import argparse

import gradio as gr

from common import BM25_INDEX_PATH, CHROMA_COLLECTION, CHROMA_DB_DIR, QUERIES_PATH, load_jsonl
from retrieve import Indexes, STRATEGIES, retrieve

MODEL_ID = "Qwen/Qwen2.5-3B"
MAX_CONTEXT_PASSAGES = 10
MAX_PASSAGE_CHARS = 1800
MAX_HISTORY_MESSAGES = 8

ARABIC_SYSTEM_PROMPT = """أنت مساعد يجيب عن أسئلة المستخدم بالاستناد إلى المقاطع المسترجعة.
أجب باللغة العربية بوضوح واختصار.
اعتمد على السياق فقط، ولا تضف معلومات من خارج المقاطع.
إذا لم يتضمن السياق إجابة كافية، فقل إن المعلومات المتاحة لا تكفي للإجابة.
أضف معرّف المقطع بين قوسين عند الاستشهاد بمعلومة.

المقاطع المسترجعة:
{context}"""


def load_generator(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the quantized Qwen model.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
    )

    model.eval()

    return tokenizer, model


def retrieved_context(indexes: Indexes, query: str, query_type: str, k_embed: int, k_rerank: int) -> str:
    _, result = retrieve(indexes, query, query_type, k_embed=k_embed, k_rerank=k_rerank)
    passage_ids = []
    if result.kind == "pages":
        for page_id, _ in result.items:
            passage_ids.extend(
                passage_id for passage_id in indexes.bm25.ids
                if indexes.meta_by_id[passage_id].get("page_id") == page_id
            )
    else:
        passage_ids = [passage_id for passage_id, _ in result.items]

    chunks = []
    for passage_id in passage_ids[:MAX_CONTEXT_PASSAGES]:
        meta = indexes.meta_by_id[passage_id]
        text = indexes.text_by_id[passage_id][:MAX_PASSAGE_CHARS]
        chunks.append(f"[معرّف المقطع: {passage_id} | الصفحة: {meta.get('page_id', '?')}]\n{text}")
    return "\n\n".join(chunks) or "لم يتم استرجاع مقاطع ذات صلة."


def generate_answer(query: str, query_type: str, history: list[dict], indexes: Indexes,
                    processor, model, k_embed: int, k_rerank: int) -> tuple[str, str]:
    context = retrieved_context(indexes, query, query_type, k_embed, k_rerank)
    messages = [{
        "role": "system",
        "content": ARABIC_SYSTEM_PROMPT.format(context=context),
    }]
    messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": query})

    print(messages)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)

    
    input_length = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    answer = processor.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    return answer, messages[0]["content"]


def create_app(model_id: str = MODEL_ID, k_embed: int = 20, k_rerank: int = 10,
               bm25_path: str = str(BM25_INDEX_PATH), chroma_dir: str = str(CHROMA_DB_DIR),
               collection: str = CHROMA_COLLECTION, queries_path: str = str(QUERIES_PATH)):
    queries = load_jsonl(queries_path)
    indexes = Indexes(bm25_path, chroma_dir, collection, use_dense=True)
    processor, model = load_generator(model_id)

    def chat(message: str, history: list[dict], query_type: str):
        answer, system_prompt = generate_answer(
            message, query_type, history, indexes, processor, model, k_embed, k_rerank
        )
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ], system_prompt

    with gr.Blocks(title="مساعد الأرشيف") as app:
        gr.Markdown("# مساعد الأرشيف\nاسأل عن المحتوى، وسيستند الجواب إلى المقاطع المسترجعة.")
        query_type = gr.Dropdown(
            choices=list(STRATEGIES), value="fact", label="نوع السؤال"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="المحادثة", rtl=True, elem_classes="rtl")
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System prompt", lines=24, interactive=False,
                    elem_classes="rtl",
                )
        message = gr.Textbox(label="سؤالك", placeholder="اكتب سؤالك هنا", elem_classes="rtl")
        with gr.Row():
            send = gr.Button("إرسال", variant="primary")
            clear = gr.Button("مسح المحادثة")

        gr.Examples(
            examples=[[row["type"], row["query"]] for row in queries if row.get("type") in STRATEGIES],
            inputs=[query_type, message],
            label="أسئلة مقترحة",
        )
        send.click(chat, inputs=[message, chatbot, query_type], outputs=[message, chatbot, system_prompt])
        message.submit(chat, inputs=[message, chatbot, query_type], outputs=[message, chatbot, system_prompt])
        clear.click(lambda: ([], "", ""), outputs=[chatbot, message, system_prompt])

    return app


def main():
    parser = argparse.ArgumentParser(description="Run the Arabic retrieval-augmented chatbot.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--k-embed", type=int, default=25)
    parser.add_argument("--k-rerank", type=int, default=10)
    parser.add_argument("--bm25-index", default=str(BM25_INDEX_PATH))
    parser.add_argument("--chroma-db", default=str(CHROMA_DB_DIR))
    parser.add_argument("--chroma-collection", default=CHROMA_COLLECTION)
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    if args.k_embed < args.k_rerank:
        parser.error("--k-embed must be greater than or equal to --k-rerank")
    app = create_app(
        model_id=args.model,
        k_embed=args.k_embed,
        k_rerank=args.k_rerank,
        bm25_path=args.bm25_index,
        chroma_dir=args.chroma_db,
        collection=args.chroma_collection,
        queries_path=args.queries,
    )
    app.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
