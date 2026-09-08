# QQ AI 聊天机器人（NoneBot2 + NapCat + DeepSeek / 智谱 GLM）

一个运行在 Windows 上的最小可用版（MVP）QQ 群聊 AI 机器人。

**当前只做一件事：**

```
群里 @机器人 你的问题  →  AI 模型回答  →  回复到当前 QQ 群
（用 DeepSeek 还是智谱 GLM，由 .env 的 AI_PROVIDER 决定）
```

## 数据流

```
QQ群成员发消息
  ↓
QQ 服务器
  ↓
NapCat（机器人账号在线，把 QQ 消息转成 OneBot 11 协议）
  ↓
反向 WebSocket：NapCat 主动连接 NoneBot2（ws://127.0.0.1:8080/onebot/v11/ws）
  ↓
NoneBot2（FastAPI 驱动，监听 127.0.0.1:8080）
  ↓
plugins/ai_chat.py（只处理群里 @机器人 的消息，提取纯文本问题）
  ↓
services/deepseek.py 或 services/zhipu.py（按 .env 的 AI_PROVIDER 选择，
AsyncOpenAI 异步调用对应模型 API）
  ↓
回答沿原路返回 → NoneBot2 → WebSocket → NapCat → QQ群回复
```

## 功能范围

已实现：

- 只处理 QQ **群聊**消息（私聊一律不响应）
- 只有 **@机器人** 才响应；没有 @ 时完全不响应
- 提取 @ 之后的纯文本问题（自动去掉 QQ 的 CQ Code）
- 调用 AI 模型 API 并回复到当前群（DeepSeek 与智谱 GLM 双服务商，`.env` 的 `AI_PROVIDER` 一键切换）
- **主备降级**：主服务商调用失败（限流、超时、Key 错误、返回为空等）时，自动改用 `AI_FALLBACK`
  指定的备用服务商重试，群里无感知
- 只 @ 不问问题时回复「有什么想问我的？」（不调用 API）
- API 异常兜底：主备都失败时回复「AI 服务暂时不可用，请稍后再试。」，Bot 不崩溃
- 启动时缺少所用服务商的 API Key（`DEEPSEEK_API_KEY` / `ZHIPU_API_KEY`）直接报错退出，而不是运行中才报错

暂不实现（保持 MVP 纯净）：上下文记忆、RAG、知识库、数据库、Redis、Function Calling、Agent、自动插话、关键词触发、私聊回复、图片理解、Web 搜索。

## 目录结构

```
qq_ai_bot/
│
├── bot.py                 # 入口：初始化 NoneBot2、注册适配器、加载插件
├── .env                   # 本地配置（含密钥，已被 gitignore，绝不提交）
├── .gitignore
├── requirements.txt
├── README.md
│
├── plugins/
│   ├── __init__.py
│   └── ai_chat.py         # 群消息处理：@ 判断 → 提取问题 → 调 AI 模型 → 回复
│
└── services/
    ├── __init__.py
    ├── deepseek.py        # DeepSeek API 客户端 + ask_deepseek(question)
    └── zhipu.py           # 智谱 GLM API 客户端 + ask_glm(question)
```

## 环境要求

- Windows 10 / 11
- Python 3.11+（本机有多个 Python 时，可用 `py -3.11` 指定版本）
- 一个普通 QQ 账号作为机器人（建议用小号）
- NapCat（本项目的 OneBot 11 接入端）
- DeepSeek API Key（https://platform.deepseek.com 申请）或
  智谱开放平台 API Key（https://open.bigmodel.cn 申请），二者按需准备一个即可

## 快速开始（Windows）

在项目根目录执行：

```powershell
# 1. 创建虚拟环境（如果本机默认 Python 不是 3.11+，请用完整路径指定，例如：
#    C:\Users\<你>\AppData\Local\Python\pythoncore-3.11-64\python.exe -m venv .venv）
python -m venv .venv

# 2. 激活虚拟环境（PowerShell）
.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 创建并编辑 .env（完整模板见下文「.env 模板」小节）
#    用 DeepSeek：AI_PROVIDER=deepseek，填 DEEPSEEK_API_KEY=sk-xxxxxxxx
#    用智谱 GLM：AI_PROVIDER=zhipu，填 ZHIPU_API_KEY=xxxxxx.xxxxxxxx

# 5. 启动 NoneBot2（先启动它，再启动 NapCat）
python bot.py

# 6. 配置并启动 NapCat（见下一节）

# 7. 在测试 QQ 群中发送： @机器人 hello
```

> 注意：必须在项目根目录运行 `python bot.py`，程序会按相对路径加载 `plugins/` 目录。

## .env 模板

项目不提供 `.env.example` 文件（避免模板文件被误提交造成配置混乱）。重建 `.env` 时，
把下面的内容复制到项目根目录的 `.env` 并填写：

```ini
# ===== NoneBot2 basic config =====
DRIVER=~fastapi
HOST=127.0.0.1
PORT=8080

# ===== AI provider =====
AI_PROVIDER=zhipu
AI_FALLBACK=deepseek

# ===== DeepSeek API =====
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash

# ===== Zhipu (GLM) API =====
ZHIPU_API_KEY=
ZHIPU_MODEL=glm-4.7-flash

# ===== OneBot access token =====
ONEBOT_ACCESS_TOKEN=
```

> `.env` 含密钥，已被 `.gitignore` 忽略，务必确认它永远不会被提交到 Git。

## AI 模型配置

机器人支持两家服务商（都是 OpenAI 兼容接口），`.env` 里的 `AI_PROVIDER` 指定**主**服务商，
`AI_FALLBACK` 指定**备用**服务商（可留空）。主服务商调用失败时自动降级到备用服务商，
群成员看到的是同一个正常回答，不感知降级过程。**修改后需重启 Bot 生效**：

| 服务商 | 配置值 | 需要填的 Key | 默认模型 | API 地址（代码中写死） |
| --- | --- | --- | --- | --- |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` | `https://api.deepseek.com` |
| 智谱 GLM | `zhipu` | `ZHIPU_API_KEY` | `glm-4.7-flash` | `https://open.bigmodel.cn/api/paas/v4` |

只有**被用到的服务商**（主 + 备用）才要求填 Key；没用到的 Key 可以留空，不影响启动。

推荐配置（GLM 平时免费，DeepSeek 兜底）：

```ini
AI_PROVIDER=zhipu      # 主：智谱 GLM
AI_FALLBACK=deepseek   # 备：GLM 限流/出错时自动改用 DeepSeek
```

### DeepSeek

```ini
AI_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_MODEL=deepseek-v4-flash
```

- Key 从 https://platform.deepseek.com 的「API Keys」页面获取。
- `DEEPSEEK_MODEL`：模型名，**不填时默认 `deepseek-v4-flash`**（代码里
  `services/deepseek.py` 的 `model = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"`）。
  DeepSeek API 当前支持：`deepseek-v4-flash`（默认，快且便宜）、`deepseek-v4-pro`（更强）、
  `deepseek-chat`（兼容别名）。注意模型名必须**全小写**，写错大小写会报 400。

### 智谱 GLM

```ini
AI_PROVIDER=zhipu
ZHIPU_API_KEY=你的智谱key
ZHIPU_MODEL=glm-4.7-flash
```

- Key 从 https://open.bigmodel.cn 的「API 密钥」页面获取（形如 `id.secret` 的完整字符串）。
- `ZHIPU_MODEL`：模型名，**不填时默认 `glm-4.7-flash`**（代码里 `services/zhipu.py`）。
  GLM-4.7-Flash 目前免费调用，具体可用模型以智谱官方文档为准。

单次请求超时默认 60 秒、网络抖动自动重试，如需调整分别改
`services/deepseek.py` / `services/zhipu.py` 里的 `*_TIMEOUT` 与 `max_retries`。

## NapCat 配置（Windows）

### 1. 安装并登录机器人账号

1. 到 https://github.com/NapNeko/NapCatQQ/releases 下载 **NapCat.Shell.zip**（Windows 推荐包，自带启动器）。
2. 解压到**纯英文路径**（例如 `D:\NapCat.Shell`，路径带中文可能出问题）。
3. **右键以管理员身份运行** `launcher.bat`，NapCat 会拉起 QQ 登录窗口。
4. 用**机器人账号**的手机 QQ 扫码登录（建议用小号，登录后保持在线）。

> 详细启动方式以官方文档为准：https://napcat.napneko.icu/guide/boot/Shell

> 本项目本机部署时已预置配置（登录后自动生效）：
> - `D:\NapCat.Shell\config\onebot11*.json`：反向 WebSocket 客户端
>   （`ws://127.0.0.1:8080/onebot/v11/ws`），并已配置与 `.env` 的 `ONEBOT_ACCESS_TOKEN`
>   一致的 Access Token；
> - `D:\NapCat.Shell\config\webui.json`：WebUI 仅监听 127.0.0.1:6099，
>   登录口令为随机生成的强口令（见该文件的 `token` 字段）。

### 2. 打开 NapCat WebUI

浏览器访问 http://127.0.0.1:6099/webui （6099 是 NapCat WebUI 默认端口）。

登录口令已预置在 `config/webui.json` 的 `token` 字段（本机随机生成，可在 WebUI 里修改）。
注意：这个口令只是登录 WebUI 用的，和 OneBot 的 Access Token 是两回事。

### 3. 创建 OneBot 11 反向 WebSocket 配置

在 WebUI 左侧进入「网络配置」，**新增一个 WebSocket 客户端**，填写：

| 字段 | 填写内容 | 说明 |
| --- | --- | --- |
| 连接地址（URL） | `ws://127.0.0.1:8080/onebot/v11/ws` | NapCat 主动连接 NoneBot2 的地址 |
| Access Token | 与 `.env` 的 `ONEBOT_ACCESS_TOKEN` 完全一致（本机已配置） | 见下方说明 |
| 消息格式 | array | 默认即可 |

保存后 NapCat 会自动连接；断线会自动重连，无需手动干预。

> 该路径 `/onebot/v11/ws` 是 nonebot-adapter-onebot 写死的反向 WebSocket 端点
> （适配器源码中同时注册了 `/onebot/v11/ws` 与 `/onebot/v11/ws/`），不要自行修改。

如果你更习惯直接改配置文件，对应的是 NapCat 目录下 `config/onebot11_<机器人QQ号>.json`：

```json
{
  "network": {
    "wsClients": [
      {
        "name": "NoneBot2",
        "url": "ws://127.0.0.1:8080/onebot/v11/ws",
        "messagePostFormat": "array",
        "reportSelfMessage": false,
        "reconnectInterval": 5000,
        "token": "与 .env 的 ONEBOT_ACCESS_TOKEN 一致"
      }
    ]
  }
}
```

### 4. 三个明确答案

- **NapCat 填什么 URL**：`ws://127.0.0.1:8080/onebot/v11/ws`
- **NoneBot2 监听哪个 HOST**：`127.0.0.1`（`.env` 的 `HOST`；NapCat 和 NoneBot2 同机时保持默认即可。
  只有 NapCat 跑在另一台机器时才改成 `0.0.0.0`）
- **NoneBot2 监听哪个 PORT**：`8080`（`.env` 的 `PORT`；改了它，NapCat 的 URL 端口也要跟着改）
- **Access Token 如何配置**：两边必须填**完全相同的字符串**（本机已配好，用于拒绝无令牌的
  WebSocket 连接，防止本机其他程序冒充 OneBot 客户端连入）：
  - NoneBot2 侧：`.env` 里的 `ONEBOT_ACCESS_TOKEN=xxx`（留空 = 不校验）
  - NapCat 侧：WebSocket 客户端配置里的 Access Token 字段（留空 = 不发送）
  - 如果只填了一边，NapCat 连接会被 NoneBot2 以 403 拒绝。

### 5. 启动顺序

先启动 NoneBot2（`python bot.py`，看到 `Uvicorn running on http://127.0.0.1:8080`），
再启动 NapCat。因为反向 WebSocket 是 **NapCat 连 NoneBot2**，NoneBot2 必须先处于监听状态。

## 测试方法（验收用例）

在测试群里，用另一个 QQ 账号操作：

| 用例 | 操作 | 预期结果 |
| --- | --- | --- |
| Case 1 | 发送 `@机器人 你好` | 机器人调用所选服务商的模型并回复 |
| Case 2 | 发送 `你好`（不 @） | 机器人**完全不响应** |
| Case 3 | 只发送 `@机器人` | 机器人回复「有什么想问我的？」，且不调用 API |
| Case 4 | 主服务商不可用（断网/限流/Key 错误） | 自动降级到备用服务商正常回答；主备都失败才回复「AI 服务暂时不可用，请稍后再试。」，Bot 不崩溃 |
| Case 5 | 连续多次正常 @ | 机器人持续正常处理每条消息 |
| Case 6 | 启动时缺少所选服务商的 API Key | 启动即报错退出，提示「未检测到 DEEPSEEK_API_KEY / ZHIPU_API_KEY」 |

> 以上用例均已在本项目开发环境中用模拟 OneBot 客户端实测通过。

### 日志参考

收到 @ 消息：

```
[INFO] ai_chat | [AI CHAT] provider=deepseek group_id=10001 user_id=20002 question=什么是STM32的DMA？
```

调用成功：

```
[INFO] ai_chat | [AI CHAT] reply success
```

调用失败（只记录异常类型与说明，绝不打印 API Key）：

```
[ERROR] services | [AI CHAT] GLM error: RateLimitError: Error code: 429 ...
[WARNING] ai_chat | [AI CHAT] 主服务商 zhipu 调用失败，降级到 deepseek 重试
[INFO] ai_chat | [AI CHAT] reply success (provider=deepseek)
```

## 安全说明

- 所有服务只监听本机回环地址（`127.0.0.1`）：NoneBot2 监听 8080、NapCat WebUI 监听 6099，
  均不暴露到公网或局域网；
- OneBot 反向 WebSocket 已配置 Access Token（`.env` 的 `ONEBOT_ACCESS_TOKEN` 与 NapCat 侧一致），
  无令牌的连接会被 403 拒绝；
- API Key 只存放在 `.env`（已被 `.gitignore` 忽略，且文件权限已收紧为仅当前用户可读写），
  错误日志经过脱敏处理，不会输出密钥内容；
- 若要把 `HOST` 改成 `0.0.0.0`（例如 NapCat 移到另一台机器），务必：
  1. 在 Windows 防火墙放行对应端口；
  2. 保持 `ONEBOT_ACCESS_TOKEN` 已设置且两边一致；
  3. 明白任何能访问该端口的机器都能向 Bot 发送消息。

## 常见问题

**Q：启动提示「未检测到 DEEPSEEK_API_KEY / ZHIPU_API_KEY」？**
当前 `AI_PROVIDER` 指向的服务商没有填 Key。要么填上对应的 Key，要么把 `AI_PROVIDER`
改成已填好 Key 的那一家，然后重启。

**Q：怎么切换 DeepSeek / 智谱 GLM？**
改 `.env`：`AI_PROVIDER=deepseek` 或 `AI_PROVIDER=zhipu`，保存后重启 `python bot.py`。

**Q：怎么设置主备降级？**
`.env` 里 `AI_PROVIDER=zhipu` + `AI_FALLBACK=deepseek` 即「平时 GLM、出错自动换 DeepSeek」。
`AI_FALLBACK` 留空表示不降级；`AI_FALLBACK` 不能与 `AI_PROVIDER` 相同。
降级在群聊中无感知，只在日志里能看到一行「主服务商 … 调用失败，降级到 … 重试」。

**Q：NapCat 一直连不上 NoneBot2？**
1. 确认 NoneBot2 已先启动并显示 `Uvicorn running on http://127.0.0.1:8080`；
2. 确认 URL 填的是 `ws://127.0.0.1:8080/onebot/v11/ws`（大小写、路径一致）；
3. 确认两边 Token 完全一致（NoneBot2 侧 `.env` 的 `ONEBOT_ACCESS_TOKEN`，NapCat 侧
   WebSocket 客户端的 Access Token）；
4. 确认 8080 端口没被其他程序占用（`netstat -ano | findstr 8080`）。

**Q：私聊机器人没反应？**
这是设计行为：当前版本只处理群聊消息。

**Q：控制台里中文乱码？**
程序已强制 stdout/stderr 用 UTF-8。如果你用的是很老的 CMD/conhost，先执行 `chcp 65001`。
使用 Windows Terminal 则无需处理。

**Q：为什么 requirements.txt / .env 的注释是英文？**
中文 Windows 的默认编码是 GBK，旧版 pip 会按 GBK 读 requirements.txt，UTF-8 中文注释会报
`UnicodeDecodeError`；pydantic 读 .env 也有类似风险。注释用英文可以从根源上避免。

**Q：想换模型？**
DeepSeek：改 `.env` 的 `DEEPSEEK_MODEL`（如 `deepseek-v4-flash`、`deepseek-v4-pro`，注意全小写）。
智谱：改 `ZHIPU_MODEL`（默认 `glm-4.7-flash`）。改完重启生效。

## 后续扩展方向（本版本不实现）

上下文记忆、RAG、知识库、Function Calling、多轮对话等。当前代码结构已按模块分离
（插件只处理消息、每个 service 只封装一家模型 API），后续扩展时在各自模块内增加即可。
