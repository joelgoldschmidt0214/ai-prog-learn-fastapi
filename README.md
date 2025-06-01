# ai-prog-learn-fastapi

AIを活用してプログラミング学習を支援するためのFastAPIバックエンドアプリケーションです。
ユーザーのレベルや目的に合わせて、解説、クイズ、参考文献などをGoogle Gemini APIを用いて生成します。

## ✨ 主な機能

* **パーソナライズされた学習支援:** ユーザーが選択したプログラミング言語、学習目的（プログラミング学習／困りごとの解決）、技術レベル、質問内容、ファイル内容に応じてAIが最適なコンテンツを生成します。
* **厳格なJSONレスポンス:** AIからの応答は、解説・クイズ・参考文献などを含む厳格なJSON形式（システムプロンプトでスキーマを強制）で返却されます。
* **ファイルアップロード対応:** ユーザーが学習に関連するファイル（テキストファイル、Jupyter Notebook等）をアップロードし、AIがその内容を文脈として活用します。
* **クイズ機能:**
  * 「プログラミング学習」目的時、AIが生成した解説に基づくクイズを1問出題。
  * ユーザーのクイズ解答をAIが採点し、マークダウン形式でフィードバックを返します（解答漏洩防止のプロンプト設計）。
* **外部AIモデル連携:** Google Gemini APIを利用してコンテンツ生成・採点を行います。
* **柔軟なプロンプト管理:** AIへの指示（ペルソナ・対話原則・出力形式・採点指示）は外部Markdownファイルで管理し、メンテナンス性とカスタマイズ性を確保しています。

## 🚀 技術スタック

* **バックエンド:** Python, FastAPI
* **AIモデル連携:** Google Generative AI (Gemini API)
* **依存ライブラリ:**
  * `uvicorn`: ASGIサーバー
  * `python-dotenv`: 環境変数の管理
  * `google-generativeai`: Gemini APIクライアント
  * `python-multipart`: FastAPIでのファイルアップロードとフォームデータ処理

## 🛠️ セットアップと実行方法

### 1. 前提条件

* Python 3.12 以上 (プロジェクトはPython 3.12.10で開発)
* pip (Pythonパッケージインストーラー)
* Git

### 2. リポジトリのクローン

```bash
git clone https://github.com/joelgoldschmidt0214/ai-prog-learn-fastapi.git
cd ai-prog-learn-fastapi
```

### 3. 仮想環境の作成と有効化 (推奨)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 4. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 5. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、Google Gemini APIキーを設定します。

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

**注意:** `.env` ファイルは `.gitignore` に追加し、APIキーがリポジトリにコミットされないようにしてください。

### 6. システムプロンプトファイルの配置

プロジェクトルートに `prompts` ディレクトリを作成し、以下のシステムプロンプトMarkdownファイルを配置してください。

* `prompts/01_persona_prompt_rev1.md` (AIのペルソナ設定)
* `prompts/02_interaction_rules_prompt_rev1.md` (AIの対話ルール・能力・倫理指針)
* `prompts/03_json_output_format_prompt.md` (AIの出力JSONフォーマット厳格指定)
* `prompts/quiz_evaluation_prompt.md` (クイズ採点用プロンプト)

**注意:** `prompts` ディレクトリも `.gitignore` に追加し、プロンプト内容がリポジトリにコミットされないようにしてください。

### 7. アプリケーションの実行

```bash
uvicorn app:app --reload
```

アプリケーションはデフォルトで `http://127.0.0.1:8000` で実行されます。

## 🌐 APIエンドポイント

主要なエンドポイントは以下の通りです。詳細なリクエスト/レスポンス形式は `app.py` 内のPydanticモデルおよびプロンプトファイルを参照してください。

* **GET `/api/gemini/status`**: Gemini APIの初期化状態を確認します。
* **POST `/api/generate`**:
  * ユーザーの入力（言語、目的、レベル、質問詳細、任意でファイル）に基づき、AIが解説・クイズ・参考文献を生成。
  * リクエストボディは `multipart/form-data` 形式。
  * レスポンスは `GeminiStructuredResponse` モデルに基づく厳格なJSON形式（システムプロンプトでスキーマ強制）。
  * クイズの解答やヒントは絶対に含まれません（プロンプトで明示的に禁止）。
* **POST `/api/evaluate-quiz`**:
  * ユーザーが送信したクイズ解答をAIが採点し、マークダウン形式でフィードバック。
  * リクエストボディは `QuizEvaluationRequest` モデル（解説・クイズ・解答を含むJSON）。
  * レスポンスは `QuizEvaluationResponse` モデル（採点コメントのみ、正解自体は返さない）。

FastAPIの自動ドキュメントも利用可能:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

## ⚙️ 設定変数

`app.py` 内で以下のグローバル変数を設定できます。

* `GEMINI_MODEL_NAME`: 使用するGeminiモデル名 (例: `'gemini-2.0-flash'`)
* `GEMINI_KNOWLEDGE_CUTOFF`: モデルの知識カットオフ日付 (例: `'2024-08'`)
* `PROMPTS_DIR`: システムプロンプトファイル格納ディレクトリ名 (デフォルト: `"prompts"`)
* `MAIN_PROMPT_FILENAMES`: メインコンテンツ生成用プロンプトファイル名タプル
* `EVALUATION_PROMPT_FILENAME`: クイズ採点用プロンプトファイル名タプル

## 📝 システムプロンプト設計のポイント

* **ペルソナ・対話原則:** AIの性格・応答方針・倫理指針を明示的に定義
* **出力フォーマット:** 厳格なJSONスキーマをプロンプトで強制し、パースエラーや解答漏洩を防止
* **クイズ採点:** 採点時も正解自体は返さず、評価コメントのみ返すようプロンプトで制御
* **ファイル活用:** アップロードファイルの内容・種類・エンコーディング等もAI応答に反映

## 🤝 コントリビューション

コントリビューションや改善提案を歓迎します！Issueを作成するか、Pull Requestを送ってください。

## 📝 ライセンス

このプロジェクトは **MIT License** の下で公開されています。
