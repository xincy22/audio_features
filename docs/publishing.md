# 发布到 PyPI

本页说明如何将 AudioFeatures 发布到 PyPI，建议先上传到 TestPyPI 进行验证。

## 发布前检查

- 确认版本号已更新（`pyproject.toml`）。
- 更新 `CHANGELOG.md`。
- 运行测试：
  - `python -m unittest discover -s tests`
- 构建包：
  - `python -m pip install -U build twine`
  - `python -m build`
- 校验构建产物：
  - `python -m twine check dist/*`

## 上传到 TestPyPI

1. 在 TestPyPI 创建 API Token。
2. 设置环境变量（PowerShell）：
   - `$env:TWINE_USERNAME="__token__"`
   - `$env:TWINE_PASSWORD="pypi-..."`
3. 上传：
   - `python -m twine upload --repository testpypi dist/*`

## 上传到 PyPI

1. 在 PyPI 创建 API Token。
2. 设置环境变量（PowerShell）：
   - `$env:TWINE_USERNAME="__token__"`
   - `$env:TWINE_PASSWORD="pypi-..."`
3. 上传：
   - `python -m twine upload dist/*`

## 发布后

- 在 git 中打 tag 并推送。
- 必要时更新文档站点。
