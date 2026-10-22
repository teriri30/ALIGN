"""发布 GitHub Release（替代 gh CLI，避免 pwsh 里 gh 参数/退出码的坑）。

用法（在 CI 里由 build-release.yml 调用）：
    python scripts/publish_release.py <zip 路径> <sha256 文件路径>

环境变量：
    GH_TOKEN            必填，GitHub token（workflow 里是 secrets.GITHUB_TOKEN）
    GITHUB_REPOSITORY   必填，形如 owner/repo
    GITHUB_REF_NAME     必填，tag 名（例如 v0.1.2）
    GITHUB_SHA          可选，目标提交

步骤：先用 generate_release_notes 建 release，再把自定义说明前置到正文，
最后逐个上传资产（大文件按 1 MiB 分块流式发送，不进内存）。
失败时打印 ::error:: 让 GitHub 直接把原因写成 check-run 注解。
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
USER_AGENT = "reui-release"


def annotate(message: str) -> None:
    safe = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title=Release::{safe}", flush=True)


def api(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail[:400]}") from exc


def upload_asset(upload_url: str, token: str, path: str) -> None:
    """流式上传，避免把几百 MB 读进内存。"""
    name = os.path.basename(path)
    query = urllib.parse.urlencode({"name": name})
    base = upload_url.split("{", 1)[0]
    parsed = urllib.parse.urlparse(base)
    size = os.path.getsize(path)
    print(f"upload {name} ({size} bytes)", flush=True)

    connection = http.client.HTTPSConnection(parsed.netloc, timeout=1800)
    connection.putrequest("POST", f"{parsed.path}?{query}")
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("Accept", "application/vnd.github+json")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Content-Length", str(size))
    connection.putheader("User-Agent", USER_AGENT)
    connection.endheaders()
    sent = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            connection.send(chunk)
            sent += len(chunk)
    response = connection.getresponse()
    body = response.read().decode("utf-8", "replace")
    connection.close()
    if response.status not in (200, 201):
        raise RuntimeError(f"upload {name} -> HTTP {response.status}: {body[:400]}")
    print(f"uploaded {name} ({sent} bytes)", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        annotate("用法: python scripts/publish_release.py <zip> <sha256>")
        return 2
    assets = [path for path in argv[:2] if path and os.path.isfile(path)]
    if not assets:
        annotate("发布资产不存在：zip 与 sha256 都没找到")
        return 2

    token = os.environ.get("GH_TOKEN") or ""
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    tag = os.environ.get("GITHUB_REF_NAME") or ""
    sha = os.environ.get("GITHUB_SHA") or ""
    if not token or not repo or not tag:
        annotate(f"缺少环境变量：token={bool(token)} repo={repo!r} tag={tag!r}")
        return 2

    try:
        version = json.load(open("build_info.json", encoding="utf-8"))["version"]
    except Exception as exc:  # noqa: BLE001 - 只用来兜底报错
        annotate(f"读不到 build_info.json 的版本号：{exc}")
        version = tag.lstrip("v")

    zip_name = os.path.basename(assets[0])
    body = (
        f"**普通用户请下载 {zip_name}（完整便携版，解压后双击 reUI.exe 即可运行，"
        "无需安装任何环境）。同名的 .sha256 文件是校验值，可不下。"
        "下方的“源代码”压缩包仅供开发者，不能直接运行。**"
    )

    try:
        payload = {
            "tag_name": tag,
            "target_commitish": sha or "main",
            "name": f"reUI {version}",
            "body": body,
            "generate_release_notes": True,
            "draft": False,
            "prerelease": False,
        }
        try:
            release = api("POST", f"{API}/repos/{repo}/releases", token, payload)
            print(f"release created: {release.get('html_url')}", flush=True)
        except RuntimeError as exc:
            if "already_exists" not in str(exc):
                raise
            print("release 已存在，改为更新", flush=True)
            release = api("GET", f"{API}/repos/{repo}/releases/tags/{tag}", token)
            release = api(
                "PATCH",
                f"{API}/repos/{repo}/releases/{release['id']}",
                token,
                {"name": payload["name"], "body": body},
            )

        upload_url = release["upload_url"]
        for path in assets:
            for attempt in range(1, 4):
                try:
                    upload_asset(upload_url, token, path)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    print(f"上传重试 {attempt}/3：{exc}", flush=True)
                    time.sleep(5)
        print(f"release published: {release.get('html_url')}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 失败必须变成注解
        annotate(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
