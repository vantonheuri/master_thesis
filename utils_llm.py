import os
import math
import json
import time
import pandas as pd
import openai

def process_llm_sentiment(
    df: pd.DataFrame,
    ticker: str,
    output_path: str,
    mode: str = "basic",   # "basic", "technical", "fundamental"
    batch_size: int = 25,
    process_all: bool = False,
    prompt_override: str = None
) -> pd.DataFrame:

    openai.api_key = os.getenv("OPENAI_API_KEY")
    df_out = df.copy()

    # Añadir columnas base si no existen
    base_cols = ["ticker_sentiment_score", "ticker_relevance_score", "ticker_price_impact"]
    if mode == "fundamental":
        base_cols += ["fundamental_score", "valuation_label"]

    for col in base_cols:
        if col not in df_out.columns:
            df_out[col] = None

    # Crear archivo si no existe
    if not os.path.exists(output_path):
        df_out[["idx"] + base_cols].to_csv(output_path, index=False)
        print("🆕 Archivo inicializado.")
    else:
        print("📁 Archivo detectado. Continuando…")

    # Prompt
    if prompt_override:
        SYSTEM_PROMPT = prompt_override
    else:
        SYSTEM_PROMPT = f"""
You are a financial news analyst tasked with evaluating the tone, relevance, and potential market impact of news articles related to a specific company (stock ticker).

For each news item, you are given:
- Date of publication
- Title
- Summary
"""
        if mode == "technical":
            SYSTEM_PROMPT += """
- Market context for that day:
  - Open, High, Low, Close, Adjusted Close prices
  - Volume traded
  - Daily return (percentage change)
"""
        if mode == "fundamental":
            SYSTEM_PROMPT += """
- Core financial fundamentals:
  - gross_margin, net_profit_margin, return_on_equity, debt_to_equity, free_cash_flow, eps
"""
        SYSTEM_PROMPT += """
Your job is to estimate:

1. `ticker_sentiment_score` → Bullish = positive, Bearish = negative. Range: -1.000 to +1.000
2. `ticker_relevance_score` → Relevance to the company fundamentals or price. Range: 0.000 to 1.000
3. `ticker_price_impact` → Expected short-term price reaction. Range: -1.000 to +1.000

Return only the following JSON format:
{
  "results": [
    {
      "idx": 123,
      "ticker_sentiment_score": 0.75,
      "ticker_relevance_score": 0.85,
      "ticker_price_impact": 0.55
    }
  ]
}
        """

    FUNCTION_DEF = {
        "name": "evaluate_news",
        "description": "Evaluates news headlines",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idx": {"type": "integer"},
                            "ticker_sentiment_score": {"type": "number"},
                            "ticker_relevance_score": {"type": "number"},
                            "ticker_price_impact": {"type": "number"},
                        },
                        "required": ["idx", "ticker_sentiment_score", "ticker_relevance_score", "ticker_price_impact"]
                    }
                }
            },
            "required": ["results"]
        }
    }

    def round_to_0_05(x):
        return round(x * 20) / 20 if pd.notnull(x) else None

    def process_batch(batch):
        lines = []
        for _, row in batch.iterrows():
            date = str(row["time_published"])
            title = str(row.get("title", "")).replace("\n", " ").strip()
            summary = str(row.get("summary", "")).replace("\n", " ").strip()[:200] + "…"
            lines.append(f"{row['idx']}|Date: {date} | Title: {title} | Summary: {summary}")

        user_content = "Evaluate the following news items:\n" + "\n".join(lines)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                functions=[FUNCTION_DEF],
                function_call={"name": "evaluate_news"},
                temperature=0.2,
                max_tokens=1500
            )

            msg = response.choices[0].message
            if msg.get("function_call"):
                args = msg.function_call.arguments
                return json.loads(args)["results"]
            else:
                print("⚠️ Respuesta sin function_call:", msg)
                return []

        except Exception as e:
            print("❌ Error OpenAI:", e)
            return []

    # Cargar CSV existente
    df_results = pd.read_csv(output_path)
    processed_idxs = set(df_results.dropna(subset=["ticker_sentiment_score"]).idx)
    remaining = df_out[~df_out["idx"].isin(processed_idxs)].copy()
    print(f"✅ Total procesadas: {len(processed_idxs)} / {len(df_out)}")

    total_batches = math.ceil(len(remaining) / batch_size)
    if not process_all:
        total_batches = 1

    for k in range(total_batches):
        start = k * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining.iloc[start:end]
        print(f"🧪 Procesando batch {k+1}/{total_batches} — filas {start} a {end-1}...")

        try:
            batch_results = process_batch(batch)
            if not batch_results:
                raise ValueError("No results returned")

            df_batch = pd.DataFrame(batch_results)
            for col in ["ticker_sentiment_score", "ticker_relevance_score", "ticker_price_impact"]:
                df_batch[col] = df_batch[col].apply(round_to_0_05)

            df_results = df_results.set_index("idx")
            df_batch = df_batch.set_index("idx")
            df_results.update(df_batch)
            df_results = df_results.reset_index()
            df_results.to_csv(output_path, index=False)

            print(f"✅ Batch {k+1} guardado correctamente.")
            time.sleep(3)

        except Exception as e:
            print(f"⚠️ Error en batch {k+1}: {e}")

    return df_results
