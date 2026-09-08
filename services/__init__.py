"""业务服务包：DeepSeek / 智谱 GLM API 封装等。"""

import os


def redact_secrets(text: str) -> str:
    """把已知的密钥（API Key、接入令牌）从文本中替换成 ***。

    用于写日志前的兜底清洗，防止任何异常信息意外把密钥带进日志。
    """
    for env_name in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "ONEBOT_ACCESS_TOKEN"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "***")
    return text
