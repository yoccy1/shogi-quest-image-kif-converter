# ShogiVision Reference

`third_party/ShogiVision/` は、ユーザーが追加した `ShogiVision-master/ShogiVision-master/` をローカル参照用にコピーしたものです。

コピー時に除外したもの:

- `venv/`
- `__pycache__/`

コピーしたもの:

- ShogiVision source/config/resources
- `models/` 配下の取得済みモデル
- `run_shogivision.ps1`

注意:

- LICENSE/モデル再配布可否は未確認。
- GitHub公開時は `.gitignore` により `third_party/ShogiVision/` を除外する。
- 既存probe結果は `reports/shogivision_probe_all_86_20260510/` にコピー済み。
