"""树莓派独立看门狗的判定逻辑测试（scripts/pi_watchdog.py）。

只测纯函数（classify / plan_actions / fmt_duration / get_config /
load_env_file / build_*_body），不触网、不重启任何服务、不发邮件。
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pi_watchdog as wd  # noqa: E402


def make_config(**overrides) -> wd.Config:
    env = {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_USER": "a@example.com",
        "SMTP_PASSWORD": "x",
        "ALERT_EMAIL_TO": "b@example.com",
    }
    env.update(overrides)
    return wd.get_config(env)


# ======================================================================
# classify：四信号 → (是否在线, 原因)
# ======================================================================


class TestClassify:
    def test_all_signals_ok_means_online(self):
        status = wd.classify(
            bot_service_active=True, internet_ok=True, napcat_login=True, onebot_link=True
        )
        assert status.online is True
        assert status.reason == "ok"

    def test_bot_service_down_is_reported_first(self):
        """qq-bot 服务没起来时，即使 NapCat 也不可达，也应先报 bot 服务。"""
        status = wd.classify(
            bot_service_active=False, internet_ok=True, napcat_login=None, onebot_link=False
        )
        assert status.online is False
        assert status.reason == "bot_service_down"

    def test_no_internet_is_reported_before_qq_state(self):
        """2026-09-28 教训：WiFi 掉线导致 QQ 失联，不能报成「去扫码」。

        那天树莓派断网 3 小时 47 分。如果只看 NapCat，就会提示用户扫码，
        把人引向完全错误的方向——问题在网络，扫码没有任何用。
        """
        status = wd.classify(
            bot_service_active=True, internet_ok=False, napcat_login=False, onebot_link=True
        )
        assert status.online is False
        assert status.reason == "no_internet"
        assert "外网" in status.detail

    def test_no_internet_does_not_trigger_restart(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        status = wd.Status(False, "no_internet", "断网")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        actions = wd.plan_actions(state, status, cfg, now=1100.0)
        assert actions == ["alert_offline"]
        assert not any(a.startswith("restart") for a in actions)

    def test_napcat_unreachable(self):
        status = wd.classify(
            bot_service_active=True, internet_ok=True, napcat_login=None, onebot_link=False
        )
        assert status.reason == "napcat_unreachable"

    def test_qq_not_logged_in(self):
        status = wd.classify(
            bot_service_active=True, internet_ok=True, napcat_login=False, onebot_link=False
        )
        assert status.reason == "qq_not_logged_in"

    def test_qq_logged_in_but_link_missing(self):
        """QQ 登录了但没有 OneBot 链路 —— 09-25 之后可能出现的形态。"""
        status = wd.classify(
            bot_service_active=True, internet_ok=True, napcat_login=True, onebot_link=False
        )
        assert status.reason == "onebot_link_missing"
        assert "8080" in status.detail

    def test_detail_uses_configured_names(self):
        status = wd.classify(
            bot_service_active=False,
            internet_ok=True,
            napcat_login=True,
            onebot_link=True,
            bot_service="my-bot",
        )
        assert "my-bot" in status.detail


# ======================================================================
# plan_actions：告警与重启决策
# ======================================================================


OFFLINE_STATUS = wd.Status(False, "qq_not_logged_in", "QQ 未登录")
BOT_DOWN_STATUS = wd.Status(False, "bot_service_down", "qq-bot 未运行")


class TestPlanActions:
    def test_online_without_history_does_nothing(self):
        cfg = make_config()
        assert wd.plan_actions({"online": True}, wd.Status(True, "ok"), cfg, now=1000.0) == []

    def test_offline_before_threshold_does_not_alert(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="300")
        state = {"offline_since": 1000.0}
        assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=1100.0) == []

    def test_offline_beyond_threshold_alerts(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="300")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=1400.0) == ["alert_offline"]

    def test_undelivered_alert_retries_every_tick(self):
        """核心保证：没送达之前，每一轮都要重试。"""
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10", WATCHDOG_REALERT_SECONDS="9999")
        state = {
            "offline_since": 1000.0,
            "alert_delivered": False,
            "last_alert_attempt_at": 1010.0,
        }
        # 上一轮刚尝试失败，这一轮（+1 秒）仍然要重试
        assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=1011.0) == ["alert_offline"]

    def test_delivered_alert_waits_for_realert_interval(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10", WATCHDOG_REALERT_SECONDS="1800")
        state = {
            "offline_since": 1000.0,
            "alert_delivered": True,
            "last_alert_delivered_at": 1010.0,
        }
        assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=1500.0) == []
        assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=2900.0) == ["alert_offline"]

    def test_delivered_alert_never_repeats_by_default(self):
        """默认（WATCHDOG_REALERT_SECONDS=0）：一次掉线只发一封。

        2026-09-29 的真实事故：默认 30 分钟一封，用户邮箱被刷爆、只能关机。
        """
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        assert cfg.realert == 0
        state = {
            "offline_since": 1000.0,
            "alert_delivered": True,
            "last_alert_delivered_at": 1010.0,
        }
        # 离线 3 天也不该再发第二封
        for t in (1500.0, 5000.0, 100000.0, 259210.0):
            assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=t) == []

    def test_undelivered_alert_still_retries_with_default_config(self):
        """『只发一封』≠『发不出去就不发了』：没送达仍然每个 tick 重试。"""
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        for t in (1100.0, 1200.0, 1300.0, 999999.0):
            assert wd.plan_actions(state, OFFLINE_STATUS, cfg, now=t) == ["alert_offline"]

    def test_brief_flap_never_emails(self):
        """几秒的抖动不发任何邮件（否则一天能刷出几十封）。"""
        cfg = make_config()  # alert_after=300
        online = wd.Status(True, "ok")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        assert wd.plan_actions(state, online, cfg, now=1100.0) == []

    def test_recovered_after_delivered_alert_notifies(self):
        cfg = make_config()
        online = wd.Status(True, "ok")
        state = {"offline_since": 1000.0, "alert_delivered": True}
        assert wd.plan_actions(state, online, cfg, now=1100.0) == ["alert_recovered"]

    def test_recovered_after_undeliverable_alert_still_notifies(self):
        """最关键的一条：断网时告警根本发不出去，恢复后必须补报。

        否则就是 09-25 的翻版——那次也是"出事了但用户一个字都没收到"。
        断网 3 小时 47 分，恢复时 alert_delivered 仍是 False。
        """
        cfg = make_config()  # alert_after=300
        online = wd.Status(True, "ok")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        # 离线 3600 秒（远超阈值）→ 必须补一封
        assert wd.plan_actions(state, online, cfg, now=4600.0) == ["alert_recovered"]

    def test_recovery_email_retries_until_it_goes_out(self):
        """补报邮件自己也可能发失败（网刚回来、SMTP 抽风）→ 每一轮都要重试。"""
        cfg = make_config()
        online = wd.Status(True, "ok")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        for t in (4600.0, 4720.0, 4840.0):  # 每 2 分钟一个 tick
            assert wd.plan_actions(state, online, cfg, now=t) == ["alert_recovered"]

    def test_recovered_body_marks_a_missed_alert(self):
        """补报邮件必须把"当时没通知出去"写清楚，否则会被当成一次普通恢复。"""
        cfg = make_config()
        normal = wd.build_recovered_body(cfg, 3600.0, "log")
        missed = wd.build_recovered_body(cfg, 3600.0, "log", missed=True)
        assert "补报" not in normal
        assert "补报" in missed
        assert "**" not in missed and "`" not in missed

    def test_qq_not_logged_in_never_restarts(self):
        """关键：扫码问题重启无用，只会让二维码失效（09-21 空转了 639 次）。"""
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        actions = wd.plan_actions(state, OFFLINE_STATUS, cfg, now=1100.0)
        assert actions == ["alert_offline"]
        assert not any(a.startswith("restart") for a in actions)

    def test_bot_service_down_triggers_restart(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        assert "restart_bot" in wd.plan_actions(state, BOT_DOWN_STATUS, cfg, now=1100.0)

    def test_restart_has_cooldown(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10", WATCHDOG_RESTART_COOLDOWN_SECONDS="1800")
        state = {
            "offline_since": 1000.0,
            "alert_delivered": False,
            "last_restart_bot_at": 1000.0,
        }
        # 冷却期内不重复重启
        assert "restart_bot" not in wd.plan_actions(state, BOT_DOWN_STATUS, cfg, now=1100.0)
        # 冷却结束后允许
        assert "restart_bot" in wd.plan_actions(state, BOT_DOWN_STATUS, cfg, now=3000.0)

    def test_napcat_unreachable_restarts_napcat(self):
        cfg = make_config(WATCHDOG_ALERT_AFTER_SECONDS="10")
        status = wd.Status(False, "napcat_unreachable", "WebUI 打不开")
        state = {"offline_since": 1000.0, "alert_delivered": False}
        assert "restart_napcat" in wd.plan_actions(state, status, cfg, now=1100.0)


# ======================================================================
# 配置与格式化
# ======================================================================


class TestConfig:
    def test_defaults(self):
        cfg = wd.get_config({})
        assert cfg.alert_after == 300
        assert cfg.realert == 0, "默认必须是『一次掉线只发一封』—— 2026-09-29 被刷爆过"
        assert cfg.restart_cooldown == 1800
        assert cfg.onebot_port == 8080
        assert cfg.can_email is False, "没配 SMTP 就不该认为能发信"

    def test_can_email_requires_all_four_fields(self):
        assert make_config().can_email is True
        assert make_config(SMTP_PASSWORD="").can_email is False
        assert make_config(ALERT_EMAIL_TO="").can_email is False

    def test_enabled_switch_parsing(self):
        assert wd.get_config({}).enabled is True
        assert wd.get_config({"WATCHDOG_ENABLED": "false"}).enabled is False
        assert wd.get_config({"WATCHDOG_ENABLED": "0"}).enabled is False

    def test_paths_are_resolved_against_project_root(self):
        cfg = wd.get_config({"WATCHDOG_LOG_FILE": "data/x.log"})
        assert cfg.log_file.is_absolute()
        assert cfg.log_file == wd.PROJECT_ROOT / "data" / "x.log"

    def test_invalid_numbers_fall_back(self):
        cfg = wd.get_config({"WATCHDOG_ALERT_AFTER_SECONDS": "abc", "SMTP_PORT": ""})
        assert cfg.alert_after == 300
        assert cfg.smtp_port == 465


class TestEnvFile:
    def test_parses_key_values_and_skips_comments(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text(
            "# comment\nSMTP_HOST=smtp.163.com\n\nSMTP_PORT=465\nQUOTED=\"v\"\n",
            encoding="utf-8",
        )
        env = wd.load_env_file(p)
        assert env["SMTP_HOST"] == "smtp.163.com"
        assert env["SMTP_PORT"] == "465"
        assert "#" not in env

    def test_missing_file_is_empty(self, tmp_path):
        assert wd.load_env_file(tmp_path / "nope") == {}


class TestFmtDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0 分"), (59, "0 分"), (60, "1 分"), (3600, "1 小时"), (90000, "1 天 1 小时")],
    )
    def test_formatting(self, seconds, expected):
        assert wd.fmt_duration(seconds).startswith(expected)


# ======================================================================
# 邮件正文：必须是纯文本，不能有 markdown 记号
# ======================================================================


def _all_bodies() -> list[tuple[str, str]]:
    """所有可能发出去的邮件正文，逐个检查。"""
    cfg = make_config()
    brief = "2026-09-28 17:46:00 [watchdog] 离线 reason=no_internet"
    bodies = []
    for reason in wd.REASON_LABELS:
        status = wd.Status(False, reason, "detail")
        bodies.append(
            (
                f"offline/{reason}",
                wd.build_offline_body(cfg, status, 3700.0, wd.build_hint(cfg, reason), brief),
            )
        )
    bodies.append(("recovered", wd.build_recovered_body(cfg, 3700.0, brief)))
    bodies.append(("recovered-empty", wd.build_recovered_body(cfg, 3700.0)))
    return bodies


class TestPlainTextBodies:
    @pytest.mark.parametrize("name,body", _all_bodies(), ids=lambda v: v if isinstance(v, str) else "")
    def test_no_markdown_markers(self, name, body):
        """纯文本邮件里 **粗体** / # 标题只会显示成字面星号井号，用户明确要求去掉。"""
        assert "**" not in body, f"{name}: 正文里还有 markdown 粗体记号"
        assert "`" not in body, f"{name}: 正文里还有反引号"
        for line in body.splitlines():
            assert not line.lstrip().startswith("#"), f"{name}: 正文里有 markdown 标题行"

    def test_hints_are_markdown_free(self):
        """提示语是拼进正文的，同样不能带记号。"""
        cfg = make_config()
        for reason in wd.HINTS:
            hint = wd.build_hint(cfg, reason)
            assert "**" not in hint, f"HINTS[{reason}] 含 **"
            assert "`" not in hint, f"HINTS[{reason}] 含反引号"

    def test_no_real_lan_ip_is_hardcoded(self):
        """仓库是公开的 —— 真实内网地址只能放树莓派本地 .env，不能进代码。"""
        import re

        source = (
            __import__("pathlib").Path(wd.__file__).read_text(encoding="utf-8")
        )
        found = re.findall(r"\b(?:192\.168|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+", source)
        assert found == [], f"pi_watchdog.py 里出现了内网 IP: {found}"

    def test_webui_hint_url_is_built_from_config(self):
        """扫码提示里的地址来自配置，不在代码里写死。"""
        cfg = make_config(WATCHDOG_WEBUI_HINT_URL="http://10.0.0.5:6099")
        assert wd.webui_hint_url(cfg) == "http://10.0.0.5:6099/webui"
        # 没配 hint 就退回 WATCHDOG_WEBUI_URL
        cfg2 = make_config(WATCHDOG_WEBUI_URL="http://127.0.0.1:6099")
        assert wd.webui_hint_url(cfg2) == "http://127.0.0.1:6099/webui"
        # 已经带 /webui 就不要重复拼
        cfg3 = make_config(WATCHDOG_WEBUI_HINT_URL="http://10.0.0.5:6099/webui")
        assert wd.webui_hint_url(cfg3) == "http://10.0.0.5:6099/webui"

    def test_hint_placeholder_is_substituted(self):
        cfg = make_config(WATCHDOG_WEBUI_HINT_URL="http://10.0.0.5:6099")
        hint = wd.build_hint(cfg, "qq_not_logged_in")
        assert "{webui}" not in hint
        assert "http://10.0.0.5:6099/webui" in hint

    def test_broken_template_never_crashes(self, monkeypatch):
        """模板写坏了也不能让告警发不出去 —— 退回原文。"""
        monkeypatch.setitem(wd.HINTS, "bot_service_down", "坏模板 {不存在的占位符}")
        assert "坏模板" in wd.build_hint(make_config(), "bot_service_down")

    def test_unknown_reason_falls_back(self):
        assert wd.build_hint(make_config(), "从没见过") == wd.DEFAULT_HINT

    def test_reason_labels_are_markdown_free(self):
        for reason, label in wd.REASON_LABELS.items():
            assert "**" not in label and "`" not in label, reason

    def test_offline_body_carries_key_facts(self):
        cfg = make_config()
        status = wd.Status(False, "no_internet", "探测目标全部不可达")
        body = wd.build_offline_body(cfg, status, 3700.0, wd.HINTS["no_internet"], "日志摘要")
        assert "1 小时" in body  # fmt_duration 的结果
        assert "no_internet" in body  # 原因代码，便于查日志
        assert "树莓派没有外网" in body  # 人类可读的原因
        assert "日志摘要" in body
        assert "附件" in body, "要告诉用户完整日志在附件里"

    def test_bodies_survive_missing_logs(self):
        cfg = make_config()
        body = wd.build_offline_body(
            cfg, wd.Status(False, "bot_service_down", "x"), 60.0, "hint", ""
        )
        assert "未采集到日志" in body
        assert "None" not in body

    def test_every_reason_has_a_label_and_a_hint(self):
        """新增 reason 时必须同时补上邮件里显示的名字与处置建议。"""
        reasons = {
            "no_internet",
            "bot_service_down",
            "napcat_unreachable",
            "qq_not_logged_in",
            "onebot_link_missing",
        }
        assert reasons <= set(wd.REASON_LABELS)
        assert reasons <= set(wd.HINTS)

    def test_restartable_reasons_are_exactly_the_expected_ones(self):
        assert set(wd.RESTART_FOR_REASON) == {
            "bot_service_down",
            "napcat_unreachable",
            "onebot_link_missing",
        }
        # 断网 / 未登录都不能重启
        assert "no_internet" not in wd.RESTART_FOR_REASON
        assert "qq_not_logged_in" not in wd.RESTART_FOR_REASON


# ======================================================================
# 掉线日志采集与附件
# ======================================================================


class TestCollectLogs:
    def test_brief_full_and_filename(self, tmp_path, monkeypatch):
        cfg = make_config(WATCHDOG_LOG_FILE=str(tmp_path / "watchdog.log"))
        cfg.log_file.write_text("line1\nline2\n", encoding="utf-8")
        monkeypatch.setattr(wd, "_journal", lambda unit, since, lines, until=None: ["journal-line"])
        brief, full, filename = wd.collect_logs(cfg, offline_since=1_700_000_000.0)

        assert filename.startswith("offline-logs-") and filename.endswith(".txt")
        assert "journal-line" in full
        assert "line1" in full
        assert "掉线现场日志" in full
        # 正文摘要不该把整份附件都塞进去
        assert len(brief) < len(full)

    def test_network_section_is_included(self, tmp_path, monkeypatch):
        """09-28 的根因在 NetworkManager，不在 napcat —— 网络层必须单独成节。"""
        cfg = make_config(WATCHDOG_LOG_FILE=str(tmp_path / "watchdog.log"))
        monkeypatch.setattr(
            wd,
            "collect_network_logs",
            lambda since, lines=wd.ATTACH_LOG_LINES: [
                "NetworkManager[9]: <info> device (wlan0): link disconnected"
            ],
        )
        monkeypatch.setattr(wd, "_journal", lambda units, since, lines: ["napcat line"])
        _, full, _ = wd.collect_logs(cfg, offline_since=1_700_000_000.0)

        assert "网络事件" in full
        assert "wlan0" in full

    def test_empty_sources_do_not_break_attachment(self, tmp_path, monkeypatch):
        cfg = make_config(WATCHDOG_LOG_FILE=str(tmp_path / "nope.log"))
        monkeypatch.setattr(wd, "_journal", lambda unit, since, lines, until=None: [])
        brief, full, filename = wd.collect_logs(cfg, offline_since=1_700_000_000.0)
        assert brief == "（未采集到日志）"
        assert "掉线现场日志" in full
        assert filename

    def test_attachment_is_size_capped(self, tmp_path, monkeypatch):
        cfg = make_config(WATCHDOG_LOG_FILE=str(tmp_path / "watchdog.log"))
        monkeypatch.setattr(wd, "_journal", lambda unit, since, lines, until=None: ["x" * 500] * 200)
        _, full, _ = wd.collect_logs(cfg, offline_since=1_700_000_000.0)
        assert len(full.encode("utf-8")) <= wd.ATTACH_MAX_BYTES + 64
        assert "已截断" in full

    def test_covers_bot_alert_log_too(self, tmp_path, monkeypatch):
        """bot 自己的投递记录（data/offline_alert.log）也要带上。

        09-25 那次就是靠它才查出"发了但没发出去"。
        """
        cfg = make_config(WATCHDOG_LOG_FILE=str(tmp_path / "watchdog.log"))
        (tmp_path / "offline_alert.log").write_text("离线告警发送失败\n", encoding="utf-8")
        monkeypatch.setattr(wd, "_journal", lambda unit, since, lines, until=None: [])
        _, full, _ = wd.collect_logs(cfg, offline_since=1_700_000_000.0)
        assert "离线告警发送失败" in full


class TestCollectNetworkLogs:
    def test_queries_network_units_directly(self, monkeypatch):
        """必须按单元直查，不能"取全系统最后 N 行再过滤"。

        本机 journal 很吵（etest / qq-bot 刷屏），实测 17:46 的 WiFi 掉线
        事件当天就已经被挤出窗口、一条都取不到了。
        """
        calls = []

        def fake_journal(units, since, lines, until=None):
            calls.append((units, until))
            return [f"{units[0]} line"] if units else []

        monkeypatch.setattr(wd, "_journal", fake_journal)
        out = wd.collect_network_logs(since=1_700_000_000.0, lines=10)

        unit_calls = [c for c in calls if c[0] is not None]
        assert len(unit_calls) == 2 * len(wd.NETWORK_UNITS), "每个单元都要查『现场』和『最近』两段"
        assert all(c[0] is not None for c in calls), "不该退化成全系统扫描"
        assert any(c[1] is not None for c in unit_calls), "有一半查询必须限定故障窗口"
        assert out

    def test_onset_window_is_captured_not_just_the_tail(self, monkeypatch):
        """09-28 离线 3h47m：只取尾部会全是 21:33 的"恢复"，看不到 17:46 的掉线。"""
        onset_call_until = []

        def fake_journal(units, since, lines, until=None):
            if units is None:
                return []
            if until is not None:
                onset_call_until.append(until)
                return ["2026-09-28T17:46:00+08:00 pi NetworkManager[9]: <info> wlan0: link disconnected"]
            return ["2026-09-28T21:33:01+08:00 pi NetworkManager[9]: <info> wlan0: activated"]

        monkeypatch.setattr(wd, "_journal", fake_journal)
        out = wd.collect_network_logs(since=1_700_000_000.0, lines=20)

        assert onset_call_until, "没有查询故障起点窗口"
        assert all(u - 1_700_000_000.0 == wd.NETWORK_ONSET_WINDOW for u in onset_call_until)
        joined = "\n".join(out)
        assert "17:46:00" in joined, "掉线起点丢了"
        assert "21:33:01" in joined, "恢复时刻丢了"
        assert out[0].startswith("2026-09-28T17:46:00"), "现场应排在最前"
        assert "中间省略" in joined

    def test_short_outage_does_not_add_a_gap_marker(self, monkeypatch):
        same = "2026-09-28T17:46:00+08:00 pi NetworkManager[9]: <info> wlan0: link disconnected"

        def fake_journal(units, since, lines, until=None):
            return [same] if units else []

        monkeypatch.setattr(wd, "_journal", fake_journal)
        out = wd.collect_network_logs(since=1_700_000_000.0, lines=20)
        assert out == [same], "两段重合时不该重复也不该插省略标记"

    def test_falls_back_to_keyword_scan(self, monkeypatch):
        """单元名对不上（netplan / iwd / connman…）时退回关键字过滤。"""

        def fake_journal(units, since, lines, until=None):
            if units:
                return []
            return [
                "iwd[9]: wlan0: disconnected",
                "unrelated spam",
            ]

        monkeypatch.setattr(wd, "_journal", fake_journal)
        out = wd.collect_network_logs(since=1_700_000_000.0, lines=10)
        assert out == ["iwd[9]: wlan0: disconnected"]

    def test_journal_filters_placeholder_lines(self, monkeypatch):
        """journalctl 的 '-- No entries --' / '-- Reboot --' 不该被当成日志。"""

        class FakeResult:
            stdout = (
                "-- No entries --\n"
                "-- Reboot --\n"
                "2026-09-28T17:46:00+08:00 pi NetworkManager[9]: <info> wlan0: link disconnected\n"
            )

        monkeypatch.setattr(wd, "_run", lambda cmd, timeout=10.0: FakeResult())
        out = wd._journal(["dhcpcd"], since=None, lines=5)
        assert out == [
            "2026-09-28T17:46:00+08:00 pi NetworkManager[9]: <info> wlan0: link disconnected"
        ]

    def test_journal_passes_multiple_units_and_until(self, monkeypatch):
        seen = {}

        class FakeResult:
            stdout = ""

        def fake_run(cmd, timeout=10.0):
            seen["cmd"] = cmd
            return FakeResult()

        monkeypatch.setattr(wd, "_run", fake_run)
        wd._journal(["a.service", "b.service"], since=1_700_000_000.0, lines=5, until=1_700_000_900.0)
        assert seen["cmd"].count("-u") == 2
        assert "--since" in seen["cmd"] and "--until" in seen["cmd"]


class TestSendEmailAttachment:
    def test_attachment_rides_along(self, monkeypatch):
        cfg = make_config()
        captured = {}

        class FakeSMTP:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def login(self, *a):
                pass

            def send_message(self, msg):
                captured["msg"] = msg

        monkeypatch.setattr(wd.smtplib, "SMTP_SSL", FakeSMTP)
        ok = wd.send_email(
            cfg,
            "[QQ Bot Alert] QQ Bot 离线",
            "纯文本正文",
            attachment=("offline-logs-1.txt", "完整日志内容"),
        )
        assert ok is True
        msg = captured["msg"]
        assert msg["Subject"] == "[QQ Bot Alert] QQ Bot 离线"
        attachments = list(msg.iter_attachments())
        assert len(attachments) == 1
        part = attachments[0]
        assert part.get_filename() == "offline-logs-1.txt"
        # 必须显式声明 charset，否则中文日志在邮件客户端里是乱码。
        # 这条同时锁死了"用 str 而不是 bytes 传 add_attachment"的实现方式。
        assert part.get_content_charset() == "utf-8"
        assert part.get_content_type() == "text/plain"
        # set_text_content 会给正文补一个结尾换行，属正常
        assert part.get_payload(decode=True).decode("utf-8").rstrip("\n") == "完整日志内容"

    def test_no_attachment_when_not_requested(self, monkeypatch):
        cfg = make_config()
        captured = {}

        class FakeSMTP:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def login(self, *a):
                pass

            def send_message(self, msg):
                captured["msg"] = msg

        monkeypatch.setattr(wd.smtplib, "SMTP_SSL", FakeSMTP)
        assert wd.send_email(cfg, "s", "b") is True
        assert list(captured["msg"].iter_attachments()) == []

    def test_without_smtp_config_nothing_is_sent(self):
        assert wd.send_email(wd.get_config({}), "s", "b") is False


class TestCheckInternet:
    def test_first_reachable_probe_wins(self, monkeypatch):
        import contextlib

        seen = []

        def fake_create_connection(target, timeout=None):
            seen.append(target)
            if target[0] == wd.INTERNET_PROBES[1][0]:
                return contextlib.nullcontext()
            raise OSError("unreachable")

        monkeypatch.setattr(wd.socket, "create_connection", fake_create_connection)
        assert wd.check_internet() is True
        assert len(seen) == 2, "第一个探测成功后就该停止"

    def test_all_probes_failing_means_offline(self, monkeypatch):
        def always_fail(target, timeout=None):
            raise OSError("unreachable")

        monkeypatch.setattr(wd.socket, "create_connection", always_fail)
        assert wd.check_internet() is False

    def test_probes_use_raw_ips(self):
        """断网时 DNS 通常一起挂 —— 用域名探测会把两种故障混在一起。"""
        for host, port in wd.INTERNET_PROBES:
            assert host[0].isdigit(), f"{host} 不是裸 IP"
            assert port == 443


# ======================================================================
# run_once：端到端串联（信号全部 mock，不触网、不发信、不重启）
# ======================================================================


class TestRunOnceRecoveryRetry:
    def _setup(self, monkeypatch, tmp_path, *, send_ok: bool, online: bool):
        cfg = make_config(
            WATCHDOG_LOG_FILE=str(tmp_path / "watchdog.log"),
            WATCHDOG_STATE_FILE=str(tmp_path / "watchdog_state.json"),
        )
        monkeypatch.setattr(wd, "check_bot_service", lambda service: True)
        monkeypatch.setattr(wd, "check_napcat_login", lambda c: True if online else False)
        monkeypatch.setattr(wd, "check_onebot_link", lambda port: True)
        monkeypatch.setattr(wd, "check_internet", lambda *a, **kw: True)
        monkeypatch.setattr(wd, "collect_logs", lambda c, since: ("摘要", "完整日志", "logs.txt"))
        sent: list[dict] = []

        def fake_send(c, subject, body, **kw):
            sent.append({"subject": subject, "body": body, **kw})
            return send_ok

        monkeypatch.setattr(wd, "send_email", fake_send)
        return cfg, sent

    def _seed_long_outage(self, cfg, *, delivered: bool):
        wd.save_state(
            cfg,
            {
                "online": False,
                "offline_since": time.time() - 4000,  # 远超 alert_after=300
                "alert_delivered": delivered,
            },
        )

    def test_missed_alert_is_reported_on_recovery(self, tmp_path, monkeypatch):
        """断网 1 小时以上、告警一封都没发出去 → 恢复时必须补报，且带日志附件。"""
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=True, online=True)
        self._seed_long_outage(cfg, delivered=False)

        assert wd.run_once(cfg) == 0

        assert len(sent) == 1
        assert "未能送出" in sent[0]["subject"]
        assert "补报" in sent[0]["body"]
        assert sent[0]["attachment"] == ("logs.txt", "完整日志")
        assert wd.load_state(cfg) == {"online": True}

    def test_failed_recovery_email_is_retried_next_tick(self, tmp_path, monkeypatch):
        """恢复邮件自己发失败时，状态必须留着，绝不能静默丢掉这次掉线。"""
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=False, online=True)
        self._seed_long_outage(cfg, delivered=False)

        wd.run_once(cfg)
        assert len(sent) == 1
        assert wd.load_state(cfg).get("offline_since"), "状态被清零 → 再也不会重试了"

        wd.run_once(cfg)
        assert len(sent) == 2, "下一个 tick 没有重试"

    def test_brief_flap_leaves_no_state(self, tmp_path, monkeypatch):
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=True, online=True)
        wd.save_state(cfg, {"online": False, "offline_since": time.time() - 5})
        wd.run_once(cfg)
        assert sent == []
        assert wd.load_state(cfg) == {"online": True}

    def test_offline_with_undelivered_alert_retries_every_tick(self, tmp_path, monkeypatch):
        """离线且发不出去 → 每个 tick 都重试（09-25 就是缺了这一步）。"""
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=False, online=False)
        self._seed_long_outage(cfg, delivered=False)

        wd.run_once(cfg)
        wd.run_once(cfg)
        assert len(sent) == 2
        assert "离线" in sent[0]["subject"]
        state = wd.load_state(cfg)
        assert state["alert_delivered"] is False
        assert state["offline_since"]  # 离线起点不能被重置

    def test_whole_outage_sends_exactly_two_emails(self, tmp_path, monkeypatch):
        """用户明确要的结果：一次掉线**只**收到两封 —— 离线 1 封 + 恢复 1 封。

        2026-09-29 之前是每 30 分钟重发，把人刷到只能关机。
        """
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=True, online=False)
        wd.save_state(cfg, {"online": False, "offline_since": time.time() - 400})

        # 第一个 tick 就该发离线告警
        wd.run_once(cfg)
        assert len(sent) == 1, f"离线期间发了 {len(sent)} 封"
        assert "离线" in sent[0]["subject"]

        # 之后 200 个 tick（≈6.5 小时）必须全部静默
        for _ in range(200):
            wd.run_once(cfg)
        assert len(sent) == 1, f"离线期间一共发了 {len(sent)} 封（应只有 1 封）"

        # 恢复了 → 第 2 封
        monkeypatch.setattr(wd, "check_napcat_login", lambda c: True)
        wd.run_once(cfg)
        assert len(sent) == 2, f"恢复后总共 {len(sent)} 封（应为 2 封）"
        assert "恢复" in sent[1]["subject"]
        assert sent[1]["attachment"], "恢复邮件必须带完整日志附件"

    def test_no_internet_email_says_network_not_qrcode(self, tmp_path, monkeypatch):
        """09-28 的误报必须彻底消失：断网时说网络，不能说"去扫码"。"""
        cfg, sent = self._setup(monkeypatch, tmp_path, send_ok=True, online=False)
        monkeypatch.setattr(wd, "check_internet", lambda *a, **kw: False)
        self._seed_long_outage(cfg, delivered=False)

        wd.run_once(cfg)

        assert len(sent) == 1
        body = sent[0]["body"]
        assert "外网" in body
        assert "树莓派没有外网" in sent[0]["subject"]
        assert "WiFi" in body
        # 断网时必须给网络类建议，而不是"去扫码"那条（09-28 用户被引错了方向）
        assert wd.HINTS["no_internet"] in body
        assert wd.HINTS["qq_not_logged_in"] not in body
