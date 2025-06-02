# app.py (google-genai SDK 完全準拠版)
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Type, Tuple, Dict

# === google-genai SDK のインポート ===
from google import genai
from google.genai import errors as genai_errors  # ★ SDK固有のエラー
import google.api_core.exceptions  # Google Cloud共通のAPIエラー

# === 型定義のインポート ===
# google.genai.types から取得 (PyPIのSafety Settings, Typesセクション参考)
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    BlockedReason,
    FinishReason,
    GoogleSearch,
    Tool,
    Part,
    Content,
    # Content, Part, Tool は google.ai.generativelanguage から取得するのがより適切
)
# from google.ai.generativelanguage import (
#     Content,  # PyPIのTypesセクションの例より
# )

import os
import json
import re
from dotenv import load_dotenv
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydanticモデル定義 (変更なし) ---
class Quiz(BaseModel):
    question: str = Field(description="クイズの質問文")


class Reference(BaseModel):
    title: str = Field(description="参考資料のタイトル")
    url: str = Field(description="参考資料のURL (有効なURL形式であること)")


class GeminiStructuredResponse(BaseModel):
    explanation: str = Field(description="主要な解説文。マークダウン形式を期待。")
    quiz: Optional[Quiz] = Field(
        None, description="目的が「プログラミング学習」の場合に生成されるクイズ"
    )
    raw_gemini_text: Optional[str] = Field(
        None,
        description="JSONパースに失敗した場合やresponse.parsedが利用できない場合のGeminiからの生テキスト",
    )


class QuizEvaluationRequest(BaseModel):
    original_explanation: str
    quiz_question: str
    user_answer: str


class QuizEvaluationResponse(BaseModel):
    evaluation_comment: str


# --- アプリケーション全体の設定変数 (変更なし) ---
GEMINI_MODEL_FOR_SOLUTION: Dict = {
    "model": "gemini-2.5-flash-preview-05-20",
    "cutoff": "2025-01",
    "max_output": 65000,
}
GEMINI_MODEL_FOR_QUIZ: Dict = {
    "model": "gemini-2.0-flash",
    "cutoff": "2024-08",
    "max_output": 8000,
}
PROMPTS_DIR = "prompts"
SOLUTION_PROMPT_FILENAMES: Tuple[str, ...] = (
    "01_persona_prompt_rev2.md",
    "02_interaction_rules_prompt_rev1.md",
    "03_json_output_format_prompt_rev1.md",
)
SOLUTION_PROMPT_FILENAMES: Tuple[str, ...] = (
    "01_persona_prompt_rev2.md",
    "02_interaction_rules_prompt_rev1.md",
    "04_grounding_prompt.md",
)
EVALUATION_PROMPT_FILENAME: Tuple[str, ...] = ("quiz_evaluation_prompt.md",)


# --- Gemini API 初期化関連 (変更なし) ---
async def initialize_gemini_model(app_instance: FastAPI):
    logger.info("Gemini APIの初期化処理を開始します (google-genai SDK)...")
    try:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("環境変数 GOOGLE_API_KEY が設定されていません。")
            app_instance.state.is_gemini_initialized = False
            app_instance.state.gemini_client = None
            return
        app_instance.state.gemini_client = genai.Client(api_key=google_api_key)
        app_instance.state.is_gemini_initialized = True
        logger.info(
            "Gemini API Client (google-genai SDK) の初期化が正常に完了しました。"
        )
    except Exception as e:
        logger.error(
            f"Gemini APIの初期化中に重大なエラーが発生しました: {e}", exc_info=True
        )
        app_instance.state.is_gemini_initialized = False
        app_instance.state.gemini_client = None
    finally:
        if hasattr(app_instance.state, "gemini_init_task"):
            app_instance.state.gemini_init_task = None


# --- Lifespanイベントハンドラ (変更なし) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (省略、前回と同じ) ...
    logger.info("FastAPIアプリケーション起動シーケンス開始 (lifespan)...")
    app.state.gemini_client = None
    app.state.is_gemini_initialized = False
    app.state.gemini_init_task = asyncio.create_task(initialize_gemini_model(app))
    logger.info(
        "Gemini初期化タスクをバックグラウンドでスケジュールしました (lifespan)。"
    )
    yield
    logger.info("FastAPIアプリケーションシャットダウンシーケンス開始 (lifespan)...")
    if hasattr(app.state, "gemini_init_task") and app.state.gemini_init_task:
        if not app.state.gemini_init_task.done():
            logger.info("実行中のGemini初期化タスクをキャンセルしようとしています...")
            app.state.gemini_init_task.cancel()
            try:
                await app.state.gemini_init_task
            except asyncio.CancelledError:
                logger.info("Gemini初期化タスクは正常にキャンセルされました。")
            except Exception as e:
                logger.error(f"Gemini初期化タスクのキャンセル中にエラー: {e}")
        else:
            logger.info("Gemini初期化タスクは既に完了していました。")


app = FastAPI(lifespan=lifespan)
# ... (CORS, root_hello, load_prompt_files, get_gemini_status は変更なし) ...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_hello():
    return {"status": "ok", "message": "FastAPI backend is operational."}


def load_prompt_files(
    prompt_filenames: Tuple[str, ...],
    add_dynamic_info: bool = True,
    model_dict: Dict = GEMINI_MODEL_FOR_SOLUTION,
) -> str:
    prompt_parts = []
    if add_dynamic_info:
        current_date_str = datetime.now().strftime("%Y年%m月%d日")
        prompt_parts.append(
            f"# システム基本情報\n- 現在の日付: {current_date_str}\n"
            f"- 利用AIモデル: {model_dict['model']}\n- AI知識カットオフ: {model_dict['cutoff']}\n"
        )
    for filename in prompt_filenames:
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompt_parts.append(f.read())
                if len(prompt_filenames) > 1 and filename != prompt_filenames[-1]:
                    prompt_parts.append("\n---\n")
            logger.info(f"プロンプトファイル '{filepath}' を読み込みました。")
        except FileNotFoundError:
            logger.error(f"プロンプトファイルが見つかりません: {filepath}")
            return f"エラー: プロンプトファイル {filepath} が見つかりません。"
        except Exception as e:
            logger.error(f"プロンプトファイル '{filepath}' の読み込み中にエラー: {e}")
            return f"エラー: プロンプトファイル {filepath} の読み込み中にエラーが発生しました。"
    return "\n".join(prompt_parts)


@app.get("/api/gemini/status")
async def get_gemini_status(request: Request):
    app_state = request.app.state

    is_initializing = False  # デフォルトはFalse
    if hasattr(app_state, "gemini_init_task") and app_state.gemini_init_task:
        if not app_state.gemini_init_task.done():
            is_initializing = True

    # initialized は initialize_gemini_model 関数内で True に設定される
    initialized_flag = (
        hasattr(app_state, "is_gemini_initialized") and app_state.is_gemini_initialized
    )

    # client_ready は client オブジェクトが実際に設定されたか
    client_is_ready = (
        hasattr(app_state, "gemini_client") and app_state.gemini_client is not None
    )

    # 最終的な準備完了状態は、タスクが完了し、かつ初期化フラグとクライアントが準備OKであること
    # ただし、タスクが完了すれば initialized_flag と client_is_ready も True になっているはず
    fully_ready = initialized_flag and client_is_ready and not is_initializing

    logger.info(
        f"Status check: is_initializing={is_initializing}, initialized_flag={initialized_flag}, client_is_ready={client_is_ready}, fully_ready={fully_ready}"
    )

    return {
        "initialized": initialized_flag,  # initialize_gemini_model の最後に True になる
        "client_ready": client_is_ready,  # client オブジェクトが設定されたか
        "is_initializing": is_initializing,  # バックグラウンドタスクがまだ実行中か
        "fully_ready_for_requests": fully_ready,  # これをフロントエンドで使うとより明確かも
    }


# parse_gemini_json_response は、response.parsed が利用できない場合のフォールバックとして残す
def parse_gemini_json_response(
    json_string: str, target_model_cls: Type[GeminiStructuredResponse]
) -> GeminiStructuredResponse:
    # ... (内容は変更なし、前回と同じ) ...
    raw_text_for_fallback = json_string
    logger.info(f"Geminiからの生の応答 (パース前): {raw_text_for_fallback}")
    try:
        match = re.search(r"```json\s*([\s\S]+?)\s*```", json_string, re.DOTALL)
        cleaned_json_string = match.group(1).strip() if match else json_string
        first_brace = cleaned_json_string.find("{")
        last_brace = cleaned_json_string.rfind("}")
        if not (
            match
            or (first_brace != -1 and last_brace != -1 and last_brace > first_brace)
        ):
            logger.error(
                f"JSON構造が見つかりませんでした。元のテキスト: {raw_text_for_fallback[:500]}..."
            )
            return GeminiStructuredResponse(
                explanation=f"エラー: Geminiからの応答を期待した形式で解析できませんでした。\n以下はモデルからの生の応答です:\n\n{raw_text_for_fallback}",
                raw_gemini_text=raw_text_for_fallback,
            )
        if not match:
            cleaned_json_string = cleaned_json_string[first_brace : last_brace + 1]
        parsed_data = json.loads(cleaned_json_string)
        logger.info(f"JSONパース成功後のデータ: {parsed_data}")
        return target_model_cls(**parsed_data)
    except json.JSONDecodeError as e_json:
        logger.error(
            f"JSONデコードエラー: {e_json}. 試行テキスト抜粋: {cleaned_json_string[:300] if 'cleaned_json_string' in locals() else raw_text_for_fallback[:300]}..."
        )
    except Exception as e_pydantic:
        logger.error(
            f"Pydanticモデルへのパース/バリデーションエラー: {e_pydantic}. パース試行データ: {parsed_data if 'parsed_data' in locals() else 'N/A'}. 元のテキスト抜粋: {raw_text_for_fallback[:300]}..."
        )
    return GeminiStructuredResponse(
        explanation=f"エラー: Geminiからの応答を期待した形式で解析できませんでした。\n以下はモデルからの生の応答です:\n\n{raw_text_for_fallback}",
        raw_gemini_text=raw_text_for_fallback,
    )


@app.post("/api/generate", response_model=GeminiStructuredResponse)
async def generate_content_with_gemini(
    request: Request,
    selected_language: str = Form(...),
    selected_goal: str = Form(...),
    selected_level: str = Form(...),
    problem_details: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    client: genai.Client = request.app.state.gemini_client
    if not client or not request.app.state.is_gemini_initialized:
        # ... (初期化チェック)
        if (
            hasattr(request.app.state, "gemini_init_task")
            and request.app.state.gemini_init_task
            and not request.app.state.gemini_init_task.done()
        ):
            raise HTTPException(status_code=503, detail="GEMINI_INITIALIZING")
        raise HTTPException(status_code=503, detail="GEMINI_UNAVAILABLE")

    # --- コンテンツパートの準備 ---
    user_text_prompt_parts = [
        f"\n# ユーザーの状況:\n- 学習/利用言語: {selected_language}\n- 現在の目的: {selected_goal}\n- 技術レベル: {selected_level}"
    ]
    if problem_details.strip():
        user_text_prompt_parts.append(
            f"\n# ユーザーからの質問や困りごと:\n{problem_details.strip()}"
        )
    else:
        # ユーザーからの入力が全くない場合は、エラーにするかデフォルトの指示を与える
        if not file:  # ファイルもなく、テキストもない場合
            logger.warning("ユーザーからの入力（テキストまたはファイル）がありません。")
            raise HTTPException(
                status_code=400, detail="質問内容またはファイルを入力してください。"
            )
        # ファイルはあるがテキストプロンプトがない場合は、ファイル処理の指示を促すようなテキストを生成
        user_text_prompt_parts.append(
            "\n# ユーザーからの具体的な質問:\n（添付ファイルを解析し、関連する情報を提供してください。）"
        )

    # file_content_for_prompt, file_info_for_prompt = "", ""
    api_contents_list = [
        Part(text="\n".join(user_text_prompt_parts))
    ]  # 少なくともテキストパートは存在

    # --- ファイル処理 (Jupyter Notebook対応と文字コード考慮) ---
    if file:
        file_name = file.filename or "uploaded_file"
        file_bytes = await file.read()
        mime_type = file.content_type or "application/octet-stream"
        logger.info(
            f"アップロードファイル '{file_name}' (MIME: {mime_type}, サイズ: {len(file_bytes)} bytes) を処理中..."
        )

        file_part_to_add = None
        file_description_for_prompt = f"\n# 添付ファイル情報:\n- ファイル名: {file_name}\n- MIMEタイプ: {mime_type}\n"

        if file_name.lower().endswith(".ipynb"):  # .ipynb ファイルの場合
            file_description_for_prompt += "種別: Jupyter Notebook\n"
            try:
                notebook_content_str = file_bytes.decode("utf-8")  # Notebookは通常UTF-8
                notebook_json = json.loads(notebook_content_str)
                code_cells = [
                    "".join(cell.get("source", []))
                    for cell in notebook_json.get("cells", [])
                    if cell.get("cell_type") == "code"
                    and cell.get("source")  # sourceが存在することも確認
                ]
                if code_cells:
                    extracted_code = "\n\n# --- コードセル区切り ---\n\n".join(
                        code_cells
                    )
                    max_chars = 30000  # プロンプトに含める最大文字数
                    if len(extracted_code) > max_chars:
                        extracted_code = extracted_code[:max_chars]
                        file_description_for_prompt += f"注意: 抽出したコード内容が長いため、先頭{max_chars}文字のみ利用します。\n"

                    # Jupyter Notebook のコードセルをテキストとして Part に含める
                    # mime_type は text/plain として扱うか、あるいは application/x-python などにするか検討
                    # ここでは抽出したコードをプレーンテキストとして扱う
                    file_part_to_add = Part.from_bytes(
                        data=extracted_code.encode("utf-8"),
                        mime_type="text/plain; charset=utf-8",
                    )
                    logger.info(
                        f"Jupyter Notebook '{file_name}' からコードセルを抽出してPartに追加しました。"
                    )
                    # プロンプトにファイル情報（抽出した旨）を追加
                    api_contents_list.append(
                        Part(
                            text=file_description_for_prompt
                            + "抽出されたコードセルを考慮して回答してください。"
                        )
                    )

                else:
                    logger.info(
                        f"Jupyter Notebook '{file_name}' にコードセルが見つかりませんでした。"
                    )
                    api_contents_list.append(
                        Part(
                            text=file_description_for_prompt
                            + "このNotebookには実行可能なコードセルが見つかりませんでした。"
                        )
                    )

            except Exception as e_ipynb:
                logger.error(
                    f"Jupyter Notebook '{file_name}' の処理中にエラー: {e_ipynb}",
                    exc_info=True,
                )
                api_contents_list.append(
                    Part(
                        text=file_description_for_prompt
                        + f"このNotebookの処理中にエラーが発生しました: {str(e_ipynb)[:100]}"
                    )
                )

        elif mime_type.startswith("text/") or any(
            file_name.lower().endswith(ext)
            for ext in [
                ".txt",
                ".md",
                ".py",
                ".js",
                ".html",
                ".css",
                ".json",
                ".csv",
                ".xml",
                ".rtf",
            ]
        ):
            # テキストベースのファイル (PDF以外)
            decoded_text = None
            detected_encoding = None
            try:
                decoded_text = file_bytes.decode("utf-8")
                detected_encoding = "UTF-8"
            except UnicodeDecodeError:
                try:
                    decoded_text = file_bytes.decode(
                        "shift-jis"
                    )  # 日本語環境でよく使われる
                    detected_encoding = "Shift-JIS"
                except UnicodeDecodeError:
                    try:
                        decoded_text = file_bytes.decode("cp932")  # Windowsの日本語
                        detected_encoding = "CP932"
                    except UnicodeDecodeError:
                        logger.warning(
                            f"ファイル '{file_name}' の自動エンコーディング判別に失敗しました。UTF-8で強制デコードを試みます。"
                        )
                        # 最終手段としてエラーを無視してデコード
                        decoded_text = file_bytes.decode("utf-8", errors="replace")
                        detected_encoding = "UTF-8 (forced, with replacements)"

            if decoded_text is not None:
                file_description_for_prompt += (
                    f"エンコーディング: {detected_encoding} (推定)\n"
                )
                max_chars = 30000
                if len(decoded_text) > max_chars:
                    decoded_text = decoded_text[:max_chars]
                    file_description_for_prompt += f"注意: ファイル内容が長いため、先頭{max_chars}文字のみ利用します。\n"

                # MIMEタイプを維持しつつ、バイトデータとしてエンコードし直してPartを作成
                # (元々テキストなら、mime_typeも text/plain などになっているはず)
                try:
                    file_part_to_add = Part.from_bytes(
                        data=decoded_text.encode(
                            detected_encoding.split(" ")[0]
                            if detected_encoding
                            else "utf-8"
                        ),
                        mime_type=mime_type
                        if mime_type.startswith("text/")
                        else "text/plain; charset=utf-8",
                    )
                    logger.info(
                        f"テキストファイル '{file_name}' をデコードしてPartに追加しました。"
                    )
                    api_contents_list.append(
                        Part(
                            text=file_description_for_prompt
                            + "添付されたテキストファイルの内容を考慮して回答してください。"
                        )
                    )
                except Exception as e_text_part:
                    logger.error(
                        f"テキストファイルのPart作成中にエラー ({file_name}): {e_text_part}",
                        exc_info=True,
                    )
                    api_contents_list.append(
                        Part(
                            text=file_description_for_prompt
                            + f"添付ファイルの処理に失敗しました: {str(e_text_part)[:100]}"
                        )
                    )

        elif mime_type == "application/pdf":
            try:
                file_part_to_add = Part.from_bytes(data=file_bytes, mime_type=mime_type)
                logger.info(f"PDFファイル '{file_name}' をPartに追加しました。")
                api_contents_list.append(
                    Part(
                        text=file_description_for_prompt
                        + "添付されたPDFファイルの内容を考慮して回答してください。"
                    )
                )
            except Exception as e_pdf_part:
                logger.error(
                    f"PDFファイルのPart作成中にエラー ({file_name}): {e_pdf_part}",
                    exc_info=True,
                )
                api_contents_list.append(
                    Part(
                        text=file_description_for_prompt
                        + f"添付PDFの処理に失敗しました: {str(e_pdf_part)[:100]}"
                    )
                )

        else:  # その他のサポートされていない可能性のあるMIMEタイプ
            logger.warning(
                f"MIMEタイプ '{mime_type}' のファイル '{file_name}' は内容の直接処理をスキップします。ファイル情報のみを渡します。"
            )
            api_contents_list.append(
                Part(
                    text=file_description_for_prompt
                    + "このファイルタイプは直接内容を解析できませんでしたが、ファイル名や種類を考慮して回答してください。"
                )
            )

        if file_part_to_add:
            api_contents_list.append(file_part_to_add)

    for i, part_obj in enumerate(api_contents_list):
        if hasattr(part_obj, "text") and part_obj.text:  # textが空でないことを確認
            logger.info(
                f"  コンテンツパート {i} (text, 抜粋): {part_obj.text[:100]}..."
            )
        elif hasattr(part_obj, "inline_data"):
            logger.info(
                f"  コンテンツパート {i} (inline_data, MIME: {part_obj.inline_data.mime_type})"
            )
    # --- Groundingツールの準備 ---
    use_grounding = selected_goal == "困りごとの解決"  # Groundingを使う条件
    tools_list_for_api = [Tool(google_search=GoogleSearch())] if use_grounding else None

    # システムプロンプトファイルの読み込み
    if use_grounding:
        system_instruction_text = load_prompt_files(
            SOLUTION_PROMPT_FILENAMES,
            add_dynamic_info=True,
            model_dict=GEMINI_MODEL_FOR_SOLUTION,
        )
        logger.info("Grounding (Google Search) ツールを有効化します。")
    else:
        system_instruction_text = load_prompt_files(
            SOLUTION_PROMPT_FILENAMES,
            add_dynamic_info=True,
            model_dict=GEMINI_MODEL_FOR_SOLUTION,
        )
    if system_instruction_text.startswith("エラー:"):
        raise HTTPException(status_code=500, detail="SYSTEM_PROMPT_LOAD_ERROR")

    logger.info(
        f"システム指示 (抜粋):\n{system_instruction_text[len(system_instruction_text) - 300 :]}..."
    )
    logger.info(f"ユーザープロンプト (抜粋):\n{user_text_prompt_parts[:300]}...")
    logger.info(f"送信するコンテンツパートの数: {len(api_contents_list)}")

    safety_settings_typed_for_api: List[SafetySetting] = [
        SafetySetting(category=hc, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
        for hc in [
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
    ]

    # --- GenerationConfig オブジェクトの構築 ---
    if use_grounding:
        # Grounding使用時は response_schema を指定しない
        generation_config_obj = GenerateContentConfig(
            response_mime_type="text/plain",  # Grounding時はプレーンテキストでJSON風文字列を受け取る
            response_schema=None,  # Grounding時はスキーマなし
            tools=tools_list_for_api,  # Grounding toolを使用
            system_instruction=Content(
                parts=[Part.from_text(text=system_instruction_text)]
            ),
            safety_settings=safety_settings_typed_for_api,
            candidate_count=1,
            # temperature=0.7, # 必要に応じて
            max_output_tokens=GEMINI_MODEL_FOR_SOLUTION["max_output"],
        )
        logger.info("Grounding利用のため、プレーンテキスト(Markdown)出力を期待します。")
    else:
        # Grounding未使用時は response_schema を指定して構造化出力を試みる
        generation_config_obj = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiStructuredResponse,
            candidate_count=1,  # 通常は1で十分
            system_instruction=Content(
                parts=[Part.from_text(text=system_instruction_text)]
            ),
            safety_settings=safety_settings_typed_for_api,
            # temperature=0.7,
            max_output_tokens=GEMINI_MODEL_FOR_QUIZ["max_output"],
        )
        logger.info(
            "Grounding未使用のため、response_schemaを設定し、application/jsonを期待します。"
        )

    model_name = (
        GEMINI_MODEL_FOR_SOLUTION["model"]
        if use_grounding
        else GEMINI_MODEL_FOR_QUIZ["model"]
    )

    try:
        logger.info(
            f"Gemini API ({model_name}) へ非同期リクエストを開始します (google-genai SDK)..."
        )
        gemini_api_response = await client.aio.models.generate_content(  # client.aio.models を使用
            model=f"models/{model_name}",  # "models/" プレフィックス
            contents=api_contents_list,  # ★ ここにテキストとファイルパートのリストが入る
            config=generation_config_obj,  # GenerateContentConfigオブジェクトを渡す
        )

        # --- レスポンスの正常性チェックとプロンプトフィードバックの確認 ---
        if (
            gemini_api_response.prompt_feedback
            and gemini_api_response.prompt_feedback.block_reason
        ):
            block_reason_str = BlockedReason(
                gemini_api_response.prompt_feedback.block_reason
            ).name
            block_message = f"BLOCKED_PROMPT: {gemini_api_response.prompt_feedback.block_reason_message or block_reason_str}"
            logger.warning(
                f"Gemini APIリクエストがブロックされました: {block_message} / Feedback: {gemini_api_response.prompt_feedback}"
            )
            raise HTTPException(status_code=400, detail=block_message)

        if not (
            gemini_api_response.candidates
            and gemini_api_response.candidates[0].content
            and gemini_api_response.candidates[0].content.parts
            and gemini_api_response.candidates[0].content.parts[0].text is not None
        ):
            logger.error(
                f"Geminiからの応答構造が予期せぬ形式です: {gemini_api_response}"
            )
            # 候補があるが finish_reason が STOP 以外の場合も考慮
            if (
                gemini_api_response.candidates
                and gemini_api_response.candidates[0].finish_reason
            ):
                finish_reason_str = FinishReason(
                    gemini_api_response.candidates[0].finish_reason
                ).name
                if finish_reason_str != "STOP":
                    logger.warning(
                        f"Gemini APIが予期せず停止しました。Finish Reason: {finish_reason_str}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"API_UNEXPECTED_STOP: Finish Reason - {finish_reason_str}",
                    )
            raise HTTPException(
                status_code=500, detail="API_INVALID_RESPONSE_STRUCTURE"
            )

        response_text_to_parse = gemini_api_response.candidates[0].content.parts[0].text

        logger.info(
            f"Geminiからの生の応答テキスト (長さ: {len(response_text_to_parse)}):\n{response_text_to_parse[:1000]}..."
        )
        # response.parsed の利用
        structured_response_obj: Optional[GeminiStructuredResponse] = None

        if use_grounding and response_text_to_parse is not None:
            # Grounding利用時は、モデルのテキスト応答をそのまま explanation に格納
            structured_response_obj = GeminiStructuredResponse(
                explanation=response_text_to_parse,
                quiz=None,  # Grounding時はクイズなし
                raw_gemini_text=response_text_to_parse,
            )
            logger.info("Grounding応答をexplanationとして設定しました。")
            if gemini_api_response.candidates[0].grounding_metadata:
                logger.info(
                    f"Grounding Metadata: {gemini_api_response.candidates[0].grounding_metadata}"
                )
        elif (
            hasattr(gemini_api_response, "parsed")
            and gemini_api_response.parsed is not None
        ):
            parsed_sdk_obj = gemini_api_response.parsed
            if isinstance(parsed_sdk_obj, GeminiStructuredResponse):
                structured_response_obj = parsed_sdk_obj
                structured_response_obj.raw_gemini_text = response_text_to_parse
                logger.info("SDK response.parsed からPydanticモデルの取得に成功。")
            elif (
                isinstance(parsed_sdk_obj, list)
                and parsed_sdk_obj
                and isinstance(parsed_sdk_obj[0], GeminiStructuredResponse)
            ):
                structured_response_obj = parsed_sdk_obj[0]
                structured_response_obj.raw_gemini_text = response_text_to_parse
                logger.info(
                    "SDK response.parsed (リスト)からPydanticモデルの取得に成功。"
                )
            else:
                logger.warning(
                    f"SDK response.parsed の型 ({type(parsed_sdk_obj)}) が期待と異なります。response_textからパースします。"
                )
        else:
            logger.warning(
                f"Geminiからの応答が期待と異なります (長さ: {len(response_text_to_parse)}):\n{response_text_to_parse}"
            )

        return structured_response_obj

    # --- 新しいエラーハンドリング階層 ---
    except (
        genai_errors.ClientError
    ) as e_client:  # google.genai.errors.ClientError (4xx系)
        # ClientError は APIError を継承しているので、APIError より先に補足
        logger.warning(
            f"Gemini API ClientError: Code={e_client.code}, Status={e_client.status}, Message={e_client.message}",
            exc_info=True,
        )
        # e_client.details に詳細なJSONが含まれる場合がある
        error_detail = (
            f"CLIENT_ERROR (Code: {e_client.code}): {e_client.message or str(e_client)}"
        )
        # プロンプトブロックは prompt_feedback で検知するが、APIレベルの400エラーもここで補足される
        status_code = e_client.code if 400 <= e_client.code < 500 else 400
        raise HTTPException(
            status_code=status_code, detail=error_detail[:200]
        )  # メッセージが長すぎないように

    except (
        genai_errors.APIError
    ) as e_sdk_api:  # google.genai.errors.APIError (より汎用的)
        logger.error(
            f"Gemini APIError: Code={e_sdk_api.code}, Status={e_sdk_api.status}, Message={e_sdk_api.message}",
            exc_info=True,
        )
        error_detail = (
            f"API_ERROR (Code: {e_sdk_api.code}): {e_sdk_api.message or str(e_sdk_api)}"
        )
        status_code = e_sdk_api.code if e_sdk_api.code else 500
        raise HTTPException(status_code=status_code, detail=error_detail[:200])

    except google.api_core.exceptions.GoogleAPIError as e_google_api:
        # これは、認証失敗、リソース上限超過、サービス利用不可など、より広範なGoogle Cloud APIエラーを補足
        logger.error(f"Google API Core Error: {e_google_api}", exc_info=True)
        # e_google_api.code() (gRPCステータスコードの場合) や e_google_api.message で詳細を取得
        status_code_grpc = e_google_api.code() if callable(e_google_api.code) else None
        error_detail = f"GOOGLE_API_CORE_ERROR (gRPC Code: {status_code_grpc}): {str(e_google_api)}"
        http_status_code = 500  # デフォルト
        if isinstance(e_google_api, google.api_core.exceptions.PermissionDenied):
            http_status_code = 403
        elif isinstance(e_google_api, google.api_core.exceptions.InvalidArgument):
            http_status_code = 400
        elif isinstance(e_google_api, google.api_core.exceptions.DeadlineExceeded):
            http_status_code = 504
        elif isinstance(e_google_api, google.api_core.exceptions.ServiceUnavailable):
            http_status_code = 503
        raise HTTPException(status_code=http_status_code, detail=error_detail[:200])

    except asyncio.TimeoutError:  # request_options の timeout
        logger.error("Gemini APIリクエストがタイムアウトしました。", exc_info=True)
        raise HTTPException(status_code=504, detail="API_TIMEOUT")

    except Exception as e_api:  # その他の予期せぬエラー
        logger.error(
            f"Gemini API呼び出しまたはレスポンス処理中に予期せぬエラー: {e_api}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"INTERNAL_SERVER_ERROR: {str(e_api)[:100]}"
        )


@app.post("/api/evaluate-quiz", response_model=QuizEvaluationResponse)
async def evaluate_quiz_answer(request_data: QuizEvaluationRequest, request: Request):
    client: genai.Client = request.app.state.gemini_client
    if not client or not request.app.state.is_gemini_initialized:
        raise HTTPException(status_code=503, detail="GEMINI_UNAVAILABLE_FOR_EVALUATION")

    evaluation_system_instruction_text = load_prompt_files(
        EVALUATION_PROMPT_FILENAME, add_dynamic_info=True
    )
    if evaluation_system_instruction_text.startswith("エラー:"):
        raise HTTPException(status_code=500, detail="EVALUATION_PROMPT_LOAD_ERROR")
    user_query_for_evaluation = f"\n# 元の解説とクイズ:\n## 解説:\n{request_data.original_explanation}\n\n## クイズの質問:\n{request_data.quiz_question}\n\n# ユーザーの解答:\n{request_data.user_answer}\n\n# あなたからのフィードバック (マークダウン形式):"

    logger.info(
        f"システム指示 (採点用抜粋):\n{evaluation_system_instruction_text[:300]}..."
    )
    logger.info(
        f"ユーザープロンプト (採点用抜粋):\n{user_query_for_evaluation[:300]}..."
    )

    safety_settings_typed_for_api: List[SafetySetting] = [
        SafetySetting(category=hc, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
        for hc in [
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
    ]

    evaluation_generation_config = GenerateContentConfig(
        temperature=0.3,  # 採点なので低めを推奨
        candidate_count=1,  # 通常は1で十分
        system_instruction=Content(
            parts=[Part.from_text(text=evaluation_system_instruction_text)]
        ),
        safety_settings=safety_settings_typed_for_api,
        max_output_tokens=GEMINI_MODEL_FOR_QUIZ["max_output"],
    )

    try:
        logger.info(
            f"Gemini ({GEMINI_MODEL_FOR_QUIZ['model']}) へクイズ採点リクエストを開始します..."
        )
        evaluation_response = await client.aio.models.generate_content(
            model=f"models/{GEMINI_MODEL_FOR_QUIZ['model']}",
            contents=[user_query_for_evaluation],
            config=evaluation_generation_config,  # オブジェクトで渡す
        )

        # --- レスポンスの正常性チェックとプロンプトフィードバックの確認 ---
        if (
            evaluation_response.prompt_feedback
            and evaluation_response.prompt_feedback.block_reason
        ):
            block_reason_str = BlockedReason(
                evaluation_response.prompt_feedback.block_reason
            ).name
            block_message = f"BLOCKED_PROMPT: {evaluation_response.prompt_feedback.block_reason_message or block_reason_str}"
            logger.warning(
                f"Gemini APIリクエストがブロックされました: {block_message} / Feedback: {evaluation_response.prompt_feedback}"
            )
            raise HTTPException(status_code=400, detail=block_message)

        if not (
            evaluation_response.candidates
            and evaluation_response.candidates[0].content
            and evaluation_response.candidates[0].content.parts
            and evaluation_response.candidates[0].content.parts[0].text
        ):
            logger.error(
                f"Geminiからの応答構造が予期せぬ形式です: {evaluation_response}"
            )
            # 候補があるが finish_reason が STOP 以外の場合も考慮
            if (
                evaluation_response.candidates
                and evaluation_response.candidates[0].finish_reason
            ):
                finish_reason_str = FinishReason(
                    evaluation_response.candidates[0].finish_reason
                ).name
                if finish_reason_str != "STOP":
                    logger.warning(
                        f"Gemini APIが予期せず停止しました。Finish Reason: {finish_reason_str}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"API_UNEXPECTED_STOP: Finish Reason - {finish_reason_str}",
                    )
            raise HTTPException(
                status_code=500, detail="API_INVALID_RESPONSE_STRUCTURE"
            )

        response_text_to_parse_eval = (
            evaluation_response.candidates[0].content.parts[0].text
        )

        logger.info(f"Geminiからの採点結果: {response_text_to_parse_eval[:200]}...")
        return QuizEvaluationResponse(evaluation_comment=response_text_to_parse_eval)

    except (
        genai_errors.ClientError
    ) as e_client:  # google.genai.errors.ClientError (4xx系)
        # ClientError は APIError を継承しているので、APIError より先に補足
        logger.warning(
            f"Gemini API ClientError: Code={e_client.code}, Status={e_client.status}, Message={e_client.message}",
            exc_info=True,
        )
        # e_client.details に詳細なJSONが含まれる場合がある
        error_detail = (
            f"CLIENT_ERROR (Code: {e_client.code}): {e_client.message or str(e_client)}"
        )
        # プロンプトブロックは prompt_feedback で検知するが、APIレベルの400エラーもここで補足される
        status_code = e_client.code if 400 <= e_client.code < 500 else 400
        raise HTTPException(
            status_code=status_code, detail=error_detail[:200]
        )  # メッセージが長すぎないように

    except (
        genai_errors.APIError
    ) as e_sdk_api:  # google.genai.errors.APIError (より汎用的)
        logger.error(
            f"Gemini APIError: Code={e_sdk_api.code}, Status={e_sdk_api.status}, Message={e_sdk_api.message}",
            exc_info=True,
        )
        error_detail = (
            f"API_ERROR (Code: {e_sdk_api.code}): {e_sdk_api.message or str(e_sdk_api)}"
        )
        status_code = e_sdk_api.code if e_sdk_api.code else 500
        raise HTTPException(status_code=status_code, detail=error_detail[:200])

    except google.api_core.exceptions.GoogleAPIError as e_google_api:
        # これは、認証失敗、リソース上限超過、サービス利用不可など、より広範なGoogle Cloud APIエラーを補足
        logger.error(f"Google API Core Error: {e_google_api}", exc_info=True)
        # e_google_api.code() (gRPCステータスコードの場合) や e_google_api.message で詳細を取得
        status_code_grpc = e_google_api.code() if callable(e_google_api.code) else None
        error_detail = f"GOOGLE_API_CORE_ERROR (gRPC Code: {status_code_grpc}): {str(e_google_api)}"
        http_status_code = 500  # デフォルト
        if isinstance(e_google_api, google.api_core.exceptions.PermissionDenied):
            http_status_code = 403
        elif isinstance(e_google_api, google.api_core.exceptions.InvalidArgument):
            http_status_code = 400
        elif isinstance(e_google_api, google.api_core.exceptions.DeadlineExceeded):
            http_status_code = 504
        elif isinstance(e_google_api, google.api_core.exceptions.ServiceUnavailable):
            http_status_code = 503
        raise HTTPException(status_code=http_status_code, detail=error_detail[:200])

    except asyncio.TimeoutError:  # request_options の timeout
        logger.error("Gemini APIリクエストがタイムアウトしました。", exc_info=True)
        raise HTTPException(status_code=504, detail="API_TIMEOUT")

    except Exception as e_api:  # その他の予期せぬエラー
        logger.error(
            f"Gemini API呼び出しまたはレスポンス処理中に予期せぬエラー: {e_api}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"INTERNAL_SERVER_ERROR: {str(e_api)[:100]}"
        )
