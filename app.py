# app.py
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
from typing import List, Optional, Type  # Typeをインポート
import google.generativeai as genai

# from google.generativeai.types import (  # ★ 例外クラスも直接インポートするか、types. でアクセス
#     GenerationConfig,
#     BlockedPromptException,
#     StopCandidateException,
# )
import os
import json
import re
from dotenv import load_dotenv
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime  # datetimeをインポート
from typing import Tuple  # Tupleをインポート

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydanticモデル定義 ---
class Quiz(BaseModel):
    question: str = Field(description="クイズの質問文")


class Reference(BaseModel):
    title: str = Field(description="参考資料のタイトル")
    url: str = Field(description="参考資料のURL (有効なURL形式であること)")


class GeminiStructuredResponse(BaseModel):
    explanation: str = Field(description="主要な解説文。マークダウン形式を期待。")
    quiz: Optional[Quiz] = Field(
        None, description="目的が「プログラミング学習」の場合に生成されるクイズ"
    )  # デフォルトをNoneに
    references: Optional[List[Reference]] = Field(
        None, description="目的が「困りごとの解決」の場合に生成される参考文献リスト"
    )  # デフォルトをNoneに
    raw_gemini_text: Optional[str] = Field(
        None,
        description="JSONパースに失敗した場合のGeminiからの生テキスト（デバッグ用）",
    )  # OptionalかつデフォルトをNoneに


class QuizEvaluationRequest(BaseModel):
    original_explanation: str
    quiz_question: str
    user_answer: str
    # selected_language: Optional[str] = None # 必要に応じてコンテキスト情報


class QuizEvaluationResponse(BaseModel):
    evaluation_comment: str  # Geminiからの採点コメント (Markdown形式を期待)
    # is_correct: Optional[bool] = None # より厳密な正誤判定が必要な場合


# データのスキーマを定義するためのクラス (FastAPIのサンプル用)
class EchoMessage(BaseModel):
    message: str | None = None


# --- アプリケーション全体の設定変数 ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # ★モデル名をここに集約 (2025年5月時点で利用可能な実際のモデル名に置き換えてください)
# 'gemini-2.5-flash-preview-05-20'
GEMINI_KNOWLEDGE_CUTOFF = (
    "2024-08"  # ★モデルの知識カットオフをここに集約 (実際のカットオフに)
)

# --- プロンプトファイル設定 ---
PROMPTS_DIR = "prompts"

# メインのコンテンツ生成用プロンプトファイル名
MAIN_PROMPT_FILENAMES: Tuple[str, ...] = (
    "01_persona_prompt.md",
    "02_interaction_rules_prompt.md",
    "03_json_output_format_prompt.md",
)

# クイズ採点用プロンプトファイル名 (タプル型で定義)
EVALUATION_PROMPT_FILENAME: Tuple[str, ...] = (
    "quiz_evaluation_prompt.md",
)  # ファイルが1つでもタプルにする


# --- Gemini API 初期化関連 ---
async def initialize_gemini_model(app_instance: FastAPI):
    logger.info("Gemini APIの初期化処理を開始します...")
    try:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("環境変数 GOOGLE_API_KEY が設定されていません。")
            app_instance.state.is_gemini_initialized = False
            app_instance.state.gemini_model = None
            return

        genai.configure(api_key=google_api_key)
        # model_name = 'gemini-2.5-flash-preview-05-20' # ← GEMINI_MODEL_NAME を使用

        app_instance.state.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        app_instance.state.is_gemini_initialized = True
        logger.info(
            f"Gemini APIの初期化が正常に完了しました。モデル: {GEMINI_MODEL_NAME}"
        )
    except Exception as e:
        logger.error(
            f"Gemini APIの初期化中に重大なエラーが発生しました: {e}", exc_info=True
        )
        app_instance.state.is_gemini_initialized = False
        app_instance.state.gemini_model = None
    finally:
        if hasattr(app_instance.state, "gemini_init_task"):
            app_instance.state.gemini_init_task = None


# --- Lifespanイベントハンドラ ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("FastAPIアプリケーション起動シーケンス開始 (lifespan)...")
    app_instance.state.gemini_model = None
    app_instance.state.is_gemini_initialized = False
    app_instance.state.gemini_init_task = asyncio.create_task(
        initialize_gemini_model(app_instance)
    )
    logger.info(
        "Gemini初期化タスクをバックグラウンドでスケジュールしました (lifespan)。"
    )

    yield

    logger.info("FastAPIアプリケーションシャットダウンシーケンス開始 (lifespan)...")
    if (
        hasattr(app_instance.state, "gemini_init_task")
        and app_instance.state.gemini_init_task
    ):
        if not app_instance.state.gemini_init_task.done():
            logger.info("実行中のGemini初期化タスクをキャンセルしようとしています...")
            app_instance.state.gemini_init_task.cancel()
            try:
                await app_instance.state.gemini_init_task
            except asyncio.CancelledError:
                logger.info("Gemini初期化タスクは正常にキャンセルされました。")
            except Exception as e:
                logger.error(f"Gemini初期化タスクのキャンセル中にエラー: {e}")
        else:
            logger.info("Gemini初期化タスクは既に完了していました。")


# FastAPIアプリケーションインスタンスの作成
app = FastAPI(lifespan=lifespan)

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- FastAPIサンプルエンドポイント ---
@app.get("/")
def root_hello():
    return {"status": "ok", "message": "FastAPI backend is operational."}


# @app.get("/api/hello")
# def hello_world():
#     return {"greeting": "Hello! はせちゅー"}


# @app.get("/api/multiply/{id}")
# def multiply(id: float):
#     doubled_value = id * 2
#     return {"input_value": id, "doubled_value": doubled_value}


# @app.get("/api/half/{id}")
# def devided(id: float):  # 関数名: divide の方が一般的
#     halfed_value = id / 2
#     return {"input_value": id, "halfed_value": halfed_value}


# @app.get("/api/count/{id}")
# def count(id: str):
#     count_value = len(id)
#     return {"input_string": id, "count_value": count_value}


# @app.post("/api/echo")
# def echo(message: EchoMessage):
#     echo_message_content = (
#         message.message if message.message else "No message provided by client"
#     )
#     return {"echoed_content": echo_message_content}


# --- プロンプト読み込み関数 (汎用化) ---
def load_prompt_files(
    prompt_filenames: Tuple[str, ...], add_dynamic_info: bool = True
) -> str:
    """
    指定されたプロンプトファイルを読み込み、結合する。
    オプションで冒頭に動的な情報（現在日付、モデル情報）を挿入する。
    """
    prompt_parts = []

    if add_dynamic_info:
        current_date_str = datetime.now().strftime("%Y年%m月%d日")
        prompt_parts.append("# システム基本情報")
        prompt_parts.append(f"- 現在の日付: {current_date_str}")
        prompt_parts.append(f"- 利用AIモデル: {GEMINI_MODEL_NAME}")
        prompt_parts.append(f"- AI知識カットオフ: {GEMINI_KNOWLEDGE_CUTOFF}\n")

    for filename in prompt_filenames:
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompt_parts.append(f.read())
                # プロンプトファイルが複数ある場合のみ区切り線を追加
                if len(prompt_filenames) > 1 and filename != prompt_filenames[-1]:
                    prompt_parts.append("\n---\n")  # プロンプト間の区切り
            logger.info(f"プロンプトファイル '{filepath}' を読み込みました。")
        except FileNotFoundError:
            logger.error(f"プロンプトファイルが見つかりません: {filepath}")
            # ここでエラー処理を行うか、デフォルトプロンプトを使用するか検討
            # この関数からは空文字列を返すか例外を投げるなど、呼び出し元で処理が必要
            return f"エラー: プロンプトファイル {filepath} が見つかりません。"
        except Exception as e:
            logger.error(f"プロンプトファイル '{filepath}' の読み込み中にエラー: {e}")
            return f"エラー: プロンプトファイル {filepath} の読み込み中にエラーが発生しました。"

    return "\n".join(prompt_parts)


# --- Gemini関連APIエンドポイント ---
@app.get("/api/gemini/status")
async def get_gemini_status(request: Request):
    app_instance = request.app
    is_initializing = False
    if (
        hasattr(app_instance.state, "gemini_init_task")
        and app_instance.state.gemini_init_task
        and not app_instance.state.gemini_init_task.done()
    ):
        is_initializing = True
    initialized = (
        hasattr(app_instance.state, "is_gemini_initialized")
        and app_instance.state.is_gemini_initialized
    )
    model_ready = (
        hasattr(app_instance.state, "gemini_model")
        and app_instance.state.gemini_model is not None
    )

    return {
        "initialized": initialized,
        "model_ready": model_ready,
        "is_initializing": is_initializing,
    }


def parse_gemini_json_response(
    json_string: str,
    target_model_cls: Type[GeminiStructuredResponse],  # 変数名を変更 (前回提案通り)
) -> GeminiStructuredResponse:
    raw_text_for_fallback = json_string
    logger.info(
        f"Geminiからの生の応答 (パース前): {raw_text_for_fallback}"
    )  # ★生の応答をログに出力

    try:
        match = re.search(r"```json\s*([\s\S]+?)\s*```", json_string, re.DOTALL)
        if match:
            cleaned_json_string = match.group(1).strip()
        else:
            first_brace = json_string.find("{")
            last_brace = json_string.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                cleaned_json_string = json_string[first_brace : last_brace + 1]
            else:
                # JSONらしき構造が見つからない場合
                logger.error(
                    f"JSON構造が見つかりませんでした。元のテキスト: {raw_text_for_fallback[:500]}..."
                )
                return GeminiStructuredResponse(
                    explanation=f"エラー: Geminiからの応答を期待した形式で解析できませんでした。\n以下はモデルからの生の応答です:\n\n{raw_text_for_fallback}",
                    raw_gemini_text=raw_text_for_fallback,
                    quiz=None,  # フォールバック時はNoneを明示
                    references=None,  # フォールバック時はNoneを明示
                )

        parsed_data = json.loads(cleaned_json_string)
        logger.info(
            f"JSONパース成功後のデータ: {parsed_data}"
        )  # ★パース後の辞書をログに出力
        return target_model_cls(**parsed_data)
    except json.JSONDecodeError as e_json:
        logger.error(
            f"JSONデコードエラー: {e_json}. クリーンアップ試行テキスト: {cleaned_json_string[:500] if 'cleaned_json_string' in locals() else 'N/A'}... 元のテキスト抜粋: {raw_text_for_fallback[:300]}..."
        )
    except Exception as e_pydantic:  # 主にPydanticのバリデーションエラー
        logger.error(
            f"Pydanticモデルへのパース/バリデーションエラー: {e_pydantic}. パース試行データ: {parsed_data if 'parsed_data' in locals() else 'N/A'}. 元のテキスト抜粋: {raw_text_for_fallback[:300]}..."
        )

    return GeminiStructuredResponse(
        explanation=f"エラー: Geminiからの応答を期待した形式で解析できませんでした。\n以下はモデルからの生の応答です:\n\n{raw_text_for_fallback}",
        raw_gemini_text=raw_text_for_fallback,
        quiz=None,
        references=None,
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
    app_instance = request.app
    if not (
        hasattr(app_instance.state, "is_gemini_initialized")
        and app_instance.state.is_gemini_initialized
        and hasattr(app_instance.state, "gemini_model")
        and app_instance.state.gemini_model is not None
    ):
        if (
            hasattr(app_instance.state, "gemini_init_task")
            and app_instance.state.gemini_init_task
            and not app_instance.state.gemini_init_task.done()
        ):
            raise HTTPException(status_code=503, detail="GEMINI_INITIALIZING")
        raise HTTPException(status_code=503, detail="GEMINI_UNAVAILABLE")

    gemini_model = app_instance.state.gemini_model

    file_content_for_prompt: str = ""
    file_info_for_prompt: str = ""
    # (ファイル処理ロジック ... )
    if file:
        file_name = file.filename
        file_info_for_prompt = f"ユーザーがアップロードしたファイル名: {file_name}\n"
        file_bytes = await file.read()
        try:
            if file_name.endswith(".ipynb"):  # type: ignore
                file_info_for_prompt += "ファイル種類: Jupyter Notebook\n"
                notebook_content_str = file_bytes.decode("utf-8")
                notebook_json = json.loads(notebook_content_str)
                code_cells_content = [
                    "".join(cell.get("source", []))
                    for cell in notebook_json.get("cells", [])
                    if cell.get("cell_type") == "code"
                ]
                file_content_for_prompt = "\n\n# --- ipynbセル区切り ---\n\n".join(
                    code_cells_content
                )
                file_info_for_prompt += "処理: コードセルを抽出しました。\n"
            else:
                file_info_for_prompt += "ファイル種類: テキストファイル\n"
                try:
                    file_content_for_prompt = file_bytes.decode("utf-8")
                    file_info_for_prompt += (
                        "エンコーディング: UTF-8で読み込みました。\n"
                    )
                except UnicodeDecodeError:
                    try:
                        file_content_for_prompt = file_bytes.decode("shift-jis")
                        file_info_for_prompt += (
                            "エンコーディング: Shift-JISで読み込みました。\n"
                        )
                    except UnicodeDecodeError:
                        file_info_for_prompt += (
                            "エンコーディング: UTF-8, Shift-JISでのデコードに失敗。\n"
                        )
                        file_content_for_prompt = ""
        except Exception as e_file:
            logger.error(
                f"ファイル処理中にエラー ({file_name}): {e_file}", exc_info=True
            )
            file_info_for_prompt += (
                f"処理エラー: {e_file}\nファイル内容は利用できません。\n"
            )
            file_content_for_prompt = ""
        if file_content_for_prompt:
            max_chars = 30000
            if len(file_content_for_prompt) > max_chars:
                file_content_for_prompt = file_content_for_prompt[:max_chars]
                file_info_for_prompt += f"注意: ファイル内容が長いため、先頭{max_chars}文字のみ利用します。\n"
            file_info_for_prompt += f"利用文字数: {len(file_content_for_prompt)}\n"

    # --- プロンプト生成 ---
    # メインのシステムプロンプトを読み込む (動的情報あり)
    system_base_prompt = load_prompt_files(MAIN_PROMPT_FILENAMES, add_dynamic_info=True)
    if system_base_prompt.startswith(
        "エラー:"
    ):  # 読み込み失敗時の簡易エラーハンドリング
        raise HTTPException(status_code=500, detail="SYSTEM_PROMPT_LOAD_ERROR")

    user_specific_prompt_parts = [
        "\n# ユーザーの状況:",
        f"- 学習/利用言語: {selected_language}",
        f"- 現在の目的: {selected_goal}",
        f"- 技術レベル: {selected_level}",
    ]
    if problem_details.strip():
        user_specific_prompt_parts.append(
            f"\n# ユーザーからの質問や困りごと:\n{problem_details.strip()}"
        )
    else:
        default_prompt_text = ""
        if selected_goal == "困りごとの解決":
            default_prompt_text = f"{selected_language}に関する一般的な問題解決のヒントや、基本的なデバッグ方法について解説してください。"
        elif selected_goal == "プログラミング学習":
            default_prompt_text = f"{selected_language}の基本的な概念や、{selected_level}の方が次に学ぶと良いトピックについて、具体的な学習ステップとともに解説してください。"
        user_specific_prompt_parts.append(
            f"\n# ユーザーからの具体的な質問:\n（具体的な質問はありませんでした。{default_prompt_text}）"
        )

    if file_info_for_prompt:
        user_specific_prompt_parts.append(
            f"\n# ユーザーが提供したファイルに関する情報:\n{file_info_for_prompt.strip()}"
        )
        if file_content_for_prompt.strip():
            user_specific_prompt_parts.append(
                f"\n# 提供されたファイルの内容 (抜粋または全文):\n```\n{file_content_for_prompt.strip()}\n```"
            )

    # システムプロンプトとユーザ固有プロンプトを結合
    final_prompt = system_base_prompt + "\n".join(user_specific_prompt_parts)

    # 最後に、JSON出力であることの念押し（これは03_json_output_format_prompt.mdの最後に含めても良い）
    final_prompt += "\n\n上記全ての指示に従い、指定されたJSON形式で応答してください。"

    logger.info(
        f"Geminiへ送信するプロンプト (最初の500文字):\n{final_prompt[:500]}..."
    )  # 全体を確認するためにはここを調整

    try:
        # GenerationConfig: response_schema を削除し、response_mime_type を使用
        generation_config = (
            genai.types.GenerationConfig(  # ★ genai.types.GenerationConfig に修正
                response_mime_type="application/json",
            )
        )

        safety_settings = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]
        logger.info(
            f"Gemini API ({gemini_model.model_name}) へのリクエストを開始します (JSONモード)..."
        )  # ★ gemini_model.model_name

        gemini_api_response = await gemini_model.generate_content_async(
            final_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options={"timeout": 120},
        )

        # response.text をパースする
        structured_response_obj = parse_gemini_json_response(
            gemini_api_response.text, GeminiStructuredResponse
        )
        return structured_response_obj
    except (
        genai.types.BlockedPromptException
    ) as e_blocked:  # ★ genai.types.BlockedPromptException
        logger.warning(f"Gemini APIリクエストがブロックされました: {e_blocked}")
        raise HTTPException(status_code=400, detail="BLOCKED_PROMPT")
    except (
        genai.types.StopCandidateException
    ) as e_stop:  # ★ genai.types.StopCandidateException
        logger.warning(f"Gemini APIが予期せず停止しました: {e_stop}")
        partial_text = ""
        if hasattr(gemini_api_response, "text") and gemini_api_response.text:
            partial_text = gemini_api_response.text
        elif (
            hasattr(e_stop, "args") and e_stop.args and isinstance(e_stop.args[0], str)
        ):
            partial_text = e_stop.args[0]
        if partial_text:
            logger.info(f"部分的なレスポンスをパース試行: {partial_text[:300]}...")
            return parse_gemini_json_response(partial_text, GeminiStructuredResponse)
        else:
            raise HTTPException(status_code=500, detail="API_UNEXPECTED_STOP")
    except asyncio.TimeoutError:
        logger.error("Gemini APIリクエストがタイムアウトしました。")
        raise HTTPException(status_code=504, detail="API_TIMEOUT")
    except Exception as e_api:
        logger.error(
            f"Gemini API呼び出しまたはレスポンス処理中に予期せぬエラー: {e_api}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR")


# --- クイズ採点エンドポイント (プロンプト生成部分を修正) ---
@app.post("/api/evaluate-quiz", response_model=QuizEvaluationResponse)
async def evaluate_quiz_answer(request_data: QuizEvaluationRequest, request: Request):
    app_instance = request.app
    if not (
        hasattr(app_instance.state, "is_gemini_initialized")
        and app_instance.state.is_gemini_initialized
        and hasattr(app_instance.state, "gemini_model")
        and app_instance.state.gemini_model is not None
    ):
        if (
            hasattr(app_instance.state, "gemini_init_task")
            and app_instance.state.gemini_init_task
            and not app_instance.state.gemini_init_task.done()
        ):
            raise HTTPException(status_code=503, detail="GEMINI_INITIALIZING")
        raise HTTPException(status_code=503, detail="GEMINI_UNAVAILABLE")

    gemini_model = app_instance.state.gemini_model

    # 採点用のシステムプロンプトを読み込む (動的情報あり)
    evaluation_system_base_prompt = load_prompt_files(
        EVALUATION_PROMPT_FILENAME, add_dynamic_info=True
    )
    if evaluation_system_base_prompt.startswith(
        "エラー:"
    ):  # 読み込み失敗時の簡易エラーハンドリング
        raise HTTPException(status_code=500, detail="EVALUATION_PROMPT_LOAD_ERROR")

    user_query_for_evaluation = f"""
# 元の解説とクイズ:
## 解説:
{request_data.original_explanation}

## クイズの質問:
{request_data.quiz_question}

# ユーザーの解答:
{request_data.user_answer}

# あなたからのフィードバック (マークダウン形式):
"""
    final_evaluation_prompt = evaluation_system_base_prompt + user_query_for_evaluation
    logger.info(
        f"Geminiへ送信する採点プロンプト (最初の500文字):\n{final_evaluation_prompt[:500]}..."
    )

    try:
        evaluation_generation_config = (
            genai.types.GenerationConfig()
        )  # ★ genai.types.GenerationConfig

        logger.info(
            f"Gemini ({gemini_model.model_name}) へクイズ採点リクエストを開始します..."
        )  # ★ gemini_model.model_name
        evaluation_response = await gemini_model.generate_content_async(
            final_evaluation_prompt,
            generation_config=evaluation_generation_config,
        )
        # ... (以降の採点エンドポイントの処理は変更なし) ...
        evaluation_comment_text = evaluation_response.text
        logger.info(f"Geminiからの採点結果: {evaluation_comment_text[:200]}...")

        return QuizEvaluationResponse(evaluation_comment=evaluation_comment_text)

    except (
        genai.types.BlockedPromptException
    ) as e_blocked:  # ★ genai.types.BlockedPromptException
        logger.warning(f"Gemini API採点リクエストがブロックされました: {e_blocked}")
        raise HTTPException(status_code=400, detail="BLOCKED_PROMPT_EVALUATION")
    except (
        genai.types.StopCandidateException
    ) as e_stop:  # ★ genai.types.StopCandidateException
        logger.warning(f"Gemini API採点が予期せず停止しました: {e_stop}")
        partial_text = (
            evaluation_response.text
            if hasattr(evaluation_response, "text") and evaluation_response.text
            else ""
        )
        if partial_text:
            return QuizEvaluationResponse(
                evaluation_comment=f"採点が途中で終了しました: {partial_text}"
            )
        else:
            raise HTTPException(
                status_code=500, detail="API_UNEXPECTED_STOP_EVALUATION"
            )
    except asyncio.TimeoutError:
        logger.error("Gemini API採点リクエストがタイムアウトしました。")
        raise HTTPException(status_code=504, detail="API_TIMEOUT_EVALUATION")
    except Exception as e_api:
        logger.error(
            f"Gemini API採点呼び出し中に予期せぬエラー: {e_api}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR_EVALUATION")
