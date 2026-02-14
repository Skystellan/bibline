# bibline

[English README](README.md)

`bibline` 是一个面向 CS 论文写作的 BibTeX 工具集：

- `rebib.py`：基于标识符优先（DOI/arXiv 等）的清洗与补全。
- `dedup_bib.py`：多协作者多个 `.bib` 的去重合并。

## 使用 `uv` 管理依赖

### 1) 安装依赖

```bash
uv sync
```

依赖来源是仓库内 `pyproject.toml`。

### 2) 运行清洗

```bash
uv run python rebib.py references_a.bib references_b.bib -o cleaned
```

### 3) 运行去重合并

```bash
uv run python dedup_bib.py alice.bib bob.bib charlie.bib \
  -o dedup_merged.bib \
  --report dedup_report.jsonl \
  --keymap dedup_keymap.json
```

## 可选：配置 Zsh 快捷命令

如果你使用 `zsh`，可以把命令写进 `~/.zshrc`，之后在任意目录直接运行。

### 方案 A：两个 alias

把下面内容加入 `~/.zshrc`：

```zsh
alias rebiber='uv run --project /absolute/path/to/bibline python /absolute/path/to/bibline/rebib.py'
alias rebibdedup='uv run --project /absolute/path/to/bibline python /absolute/path/to/bibline/dedup_bib.py'
```

按本仓库路径的示例：

```zsh
alias rebiber='uv run --project /Users/skystellan/ResearchTools/Rebiber python /Users/skystellan/ResearchTools/Rebiber/rebib.py'
alias rebibdedup='uv run --project /Users/skystellan/ResearchTools/Rebiber python /Users/skystellan/ResearchTools/Rebiber/dedup_bib.py'
```

### 方案 B：统一成一个命令 `rebiber`

把下面函数加入 `~/.zshrc`：

```zsh
rebiber() {
  local proj="/absolute/path/to/bibline"
  if [[ "$1" == "dedup" ]]; then
    shift
    uv run --project "$proj" python "$proj/dedup_bib.py" "$@"
  else
    uv run --project "$proj" python "$proj/rebib.py" "$@"
  fi
}
```

然后执行：

```bash
source ~/.zshrc
```

使用示例：

```bash
rebiber refs1.bib refs2.bib -o cleaned
rebiber dedup alice.bib bob.bib -o dedup_merged.bib
```

## `rebib.py` 功能

- 解析优先级：DOI -> arXiv DOI 升级 -> DBLP -> Crossref/OpenAlex/arXiv 回退。
- 默认保持 citekey（`ID`）不变。
- venue-aware：已发表条目不会被降级成纯 arXiv 记录。
- 标题大写保护采用保守策略（缩写词、混合大小写词等）。
- 默认保留 URL 便于人工核查。
- 默认支持跨文件同步（`--no-sync` 可关闭）。
- 输出 `rebib_report.jsonl` 便于审计。
- 支持进度显示与速率调优：
  - `--workers`
  - `--http-timeout`
  - `--http-retries`
  - `--http-backoff`
  - `--no-cache`
  - `--no-progress`

## `dedup_bib.py` 功能

- 强匹配聚类：`doi` / `arXiv` / `OpenReview`。
- 无标识符时用 `标题 + 第一作者姓 + 年份` 进行桥接去重。
- 聚类代表条目优先选择已有 venue 的记录，再按完整度选择。
- 输出：
  - 合并去重后的 `.bib`
  - `dedup_report.jsonl`（聚类详情）
  - `dedup_keymap.json`（旧 key 到 canonical key 映射）

## 推荐协作流程

1. 每位协作者维护自己的 `.bib`。
2. 先对每份 `.bib` 运行 `rebib.py` 做清洗。
3. 再对全部 `.bib` 运行 `dedup_bib.py` 做全局去重合并。
4. 如需统一引用 key，用 `dedup_keymap.json` 批量替换文稿中的引用键。

## 命令参考

### `rebib.py`

```bash
uv run python rebib.py [-h] [--inplace] [-o OUTDIR] [--report REPORT] [--no-sync]
                       [--drop-url-with-doi] [--workers WORKERS]
                       [--http-timeout HTTP_TIMEOUT] [--http-retries HTTP_RETRIES]
                       [--http-backoff HTTP_BACKOFF] [--no-cache] [--no-progress]
                       inputs [inputs ...]
```

### `dedup_bib.py`

```bash
uv run python dedup_bib.py [-h] [-o OUTPUT] [--report REPORT] [--keymap KEYMAP]
                           [--quiet]
                           inputs [inputs ...]
```

## 安全提示

不要将 API Key 或 token 直接提交到仓库。请使用环境变量。
