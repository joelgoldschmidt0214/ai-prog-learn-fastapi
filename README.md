# ai-prog-learn-fastapi

AIを活用してプログラミング学習を支援するためのFastAPIバックエンドアプリケーションです。
ユーザーのレベルや目的に合わせて、解説、クイズ、参考文献などを提供します。

## ✨ 主な機能

* **パーソナライズされた学習支援:** ユーザーが選択したプログラミング言語、学習目的（プログラミング学習／困りごとの解決）、技術レベルに応じて、AIが最適なコンテンツを生成します。
* **構造化されたAIレスポンス:** AIからの応答は、解説、クイズ、参考文献といった要素を含むJSON形式で返却されます。
* **ファイルアップロード対応:** ユーザーが学習に関連するファイル（テキストファイル、Jupyter Notebookなど）をアップロードし、AIがその内容を理解して回答に活用できます。
* **クイズ機能:**
  * 「プログラミング学習」目的の場合、AIが生成した解説に基づいて理解度を確認するためのクイズが出題されます。
  * ユーザーがクイズの解答を送信すると、AIが採点し、フィードバックを返します。
* **外部AIモデル連携:** GoogleのGemini APIを利用してコンテンツ生成を行います。
* **柔軟なプロンプト管理:** AIへの指示（システムプロンプト）は外部Markdownファイルで管理されており、メンテナンス性とカスタマイズ性が向上しています。

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

* Python 3.11 以上 (プロジェクトはPython 3.12.10で開発されました)
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
source .venv/bin/activate  # Linux/macOS の場合
# .venv\Scripts\activate    # Windows の場合
```

### 4. 依存関係のインストール

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 5. 環境変数の設定

プロジェクトのルートディレクトリに `.env` ファイルを作成し、Google Gemini APIキーを設定します。

```env
# .env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

**注意:** `.env` ファイルは `.gitignore` に追加して、APIキーがリポジトリにコミットされないようにしてください。

### 6. システムプロンプトファイルの配置

プロジェクトルートに `prompts` ディレクトリを作成し、以下のシステムプロンプト用Markdownファイルを配置します。これらのファイルの内容は、AIの挙動を定義する上で非常に重要です。

* `prompts/01_persona_prompt.md` (AIのペルソナ設定)
* `prompts/02_interaction_rules_prompt.md` (AIの対話ルールと能力)
* `prompts/03_json_output_format_prompt.md` (AIの出力JSONフォーマット指定)
* `prompts/quiz_evaluation_prompt.md` (クイズ採点用プロンプト)

**注意:** `prompts` ディレクトリも `.gitignore` に追加して、プロンプトの内容がリポジトリにコミットされないようにすることを推奨します。

### 7. アプリケーションの実行

Uvicornを使用してFastAPIアプリケーションを起動します。

```bash
uvicorn app:app --reload
```

`--reload` オプションは開発中に便利で、コード変更時にサーバーを自動的に再起動します。

アプリケーションはデフォルトで `http://127.0.0.1:8000` で実行されます。

## 🌐 APIエンドポイント

主要なエンドポイントは以下の通りです。詳細なリクエスト/レスポンス形式はソースコード (`app.py`内のPydanticモデル) を参照してください。

* **GET `/api/gemini/status`**: Gemini APIの初期化状態を確認します。
* **POST `/api/generate`**:
  * ユーザーの入力（言語、目的、レベル、質問詳細、任意でファイル）に基づいて、AIが解説、クイズ、または参考文献を生成します。
  * リクエストボディは `multipart/form-data` 形式です。
  * レスポンスは `GeminiStructuredResponse` モデルに基づいたJSON形式です。
* **POST `/api/evaluate-quiz`**:
  * ユーザーが送信したクイズの解答をAIが採点し、フィードバックを返します。
  * リクエストボディは `QuizEvaluationRequest` モデルに基づいたJSON形式です。
  * レスポンスは `QuizEvaluationResponse` モデルに基づいたJSON形式です。

FastAPIが提供する自動ドキュメンテーションも利用可能です:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

## ⚙️ 設定変数

`app.py` 内で以下のグローバル変数を設定できます。

* `GEMINI_MODEL_NAME`: 使用するGeminiモデル名 (例: `'gemini-2.0-flash'`)
* `GEMINI_KNOWLEDGE_CUTOFF`: 使用するモデルの知識カットオフ日付 (例: `'2024-08'`)
* `PROMPTS_DIR`: システムプロンプトファイルが格納されているディレクトリ名 (デフォルト: `"prompts"`)
* `MAIN_PROMPT_FILENAMES`: メインのコンテンツ生成に使用するプロンプトファイル名のタプル
* `EVALUATION_PROMPT_FILENAME`: クイズ採点に使用するプロンプトファイル名のタプル

## 🤝 コントリビューション

コントリビューションや改善提案を歓迎します！Issueを作成するか、Pull Requestを送ってください。

## 📝 ライセンス

このプロジェクトは **MIT License** の下で公開されています。
