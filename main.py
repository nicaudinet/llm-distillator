import argparse
import asyncio
import ssl
import time
from enum import Enum
from itertools import batched
from pathlib import Path

import aiohttp
import certifi
import ollama
from datasets import load_dataset
from dotenv import dotenv_values
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
)


def parse_amazon_review(line: dict) -> dict:
    split_line = line["text"].split(" ")
    return {"text": " ".join(split_line[3:])}


def make_amazon_prompt(text: str) -> str:
    prompt = f"""
Rewrite the review such that the sentiment is completely neutral. It is very
important that one cannot tell whether the review is positive or negative at
all. Try and keep all other information in the review.

Here are a few examples of how to do this.

Example 1: if the original review was:

i bought this album because i loved the title song . it 's such a great song ,
how bad can the rest of the album be , right ? well , the rest of the songs are
just filler and are n't worth the money i paid for this . it 's either shameless
bubblegum or oversentimentalized depressing tripe . kenny chesney is a popular
artist and as a result he is in the cookie cutter category of the nashville
music scene . he 's gotta pump out the albums so the record company can keep
lining their pockets while the suckers out there keep buying this garbage to
perpetuate more garbage coming out of that town . i 'll get down off my soapbox
now . but country music really needs to get back to it 's roots and stop this
pop nonsense . what country music really is and what it is considered to be by
mainstream are two different things .

then the neutral rewrite might be:

I bought this album because of the title song. The rest of the album I didn't
know as well. Kenny Chesney is a popular artist in the Nashville music scene. He
makes many albums with his record company. Country music has been evolving from
its roots to a more pop sound.

Example 2: if the original review was:

this is a very good shaver for the private area . however , the key to getting
the best results is to trim the longer hairs with scissors or the largest guard
first . this will keep the shaver from pulling on the longer hairs and will
enable the foil part of the shaver to work . the foil will not be able to do its
job if the hairs are too long . the only problem i had with the shaver was that
it did not enable me to shave my back like it claimed . however , i use the '
mangroomer ' back shaver for this and it is perfect for you to shave off all
your back hair easily with its elongated handle . it is a great product as well
. therefore , i would have to say these two products coupled together seem to
cover all the bases for men 's grooming on the body . i would highly recommend
both of them for perfect manscaping results

then the neutral rewrite might be:

To use this shaver in the private area it is important to trim the longer hairs
with scissors or the largest guard first. This will keep the shaver from pulling
on the longer hairs and will enable the foil part of the shaver to work. The
foil will not be able to do its job if the hairs are too long. The shaver might
also not work well on the back. For this, there are other options such as the
'Mangroomer' back shaver which has an elongated handle that makes it easy to
shave back hair.

Example 3: if the original review was:

i bought bead fantasies and bead fantasies ii at the same time after reading the
positive reviews ; i wish i had looked at these books before buying . there are
pretty motifs that i will incorporate into my beading projects but i find the
small typed directions overly simplistic and the diagrams are too small . i 'm
glad this is n't my first beading book or i would feel totally discouraged from
trying any of these projects . i wo n't be buying bead fantasies iii . the art
and elegance of beadweaving and coraling technique remain my favorite beading
books .

then the neutral rewrite might be:

I bought Bead Fantasies and Bead Fantasies II at the same time. I like some of
the motifs but not others. This is not my first beading book. The art and
elegance of beadweaving and coraling technique are great beading books. 

Here's the review:

{text["text"]}
"""
    return {"text": prompt}


async def call_openai(session, prompt, model, openai_api_key):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": 300,
        "n": 1,
    }
    async with session.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        },
        json=payload,
        ssl=ssl.create_default_context(cafile=certifi.where()),
    ) as response:
        response = await response.json()
    return response


async def call_openai_bulk(prompts, model, openai_api_key):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
        responses = []
        for prompt in prompts:
            responses.append(
                tg.create_task(call_openai(session, prompt, model, openai_api_key))
            )
    return [response.result() for response in responses]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=["ollama", "alvis", "gpt4"],
        help="How to run the model",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=Path,
        help="Path to the Amazon review data file",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        help="Directory to save responses to (default: ./results)",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="The number of samples to use (default: all)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="The batch size to use for the pipeline",
    )
    args = parser.parse_args()

    # Load the data
    data = load_dataset(
        "text",  # Type of data to load (text file)
        data_files=str(args.data_path),  # Filepath for the data
        split="train",  # Return a Dataset rather than a DatasetDict
    )

    review_start = 800

    # Preprocess
    data = data.map(parse_amazon_review)
    if args.num_samples is not None:
        data = data.select(range(review_start, review_start + args.num_samples))

    # Make the prompts
    prompts = data.map(make_amazon_prompt)

    if args.runmode == "ollama":
        print("Running with Ollama")
        responses = []
        for prompt in prompts:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt["text"]}],
            )
            print(f"Prompt:\n{prompt['text']}")
            print(f"Response:\n{response['message']['content']}")

    elif args.runmode == "gpt4":
        model = "gpt-4-0125-preview"
        print(f"Running with {model}")
        # From https://medium.com/@nitin_l/parallel-chatgpt-requests-from-python-6ab48cc2a610
        env = dotenv_values(".env")
        batch_size = 100  # There's a 500 RPM limit
        for i, prompts_batch in enumerate(batched(prompts, batch_size)):
            print(f"Batch {i}")
            start = time.time()
            responses = asyncio.run(
                call_openai_bulk(
                    prompts=prompts_batch,
                    model=model,
                    openai_api_key=env["OPENAI_API_KEY"],
                )
            )
            for j, response in enumerate(responses):
                review_id = review_start + i * batch_size + j
                with open(
                    args.out_dir / Path(f"{str(review_id).zfill(5)}.txt"), "w"
                ) as f:
                    f.write(response["choices"][0]["message"]["content"])
            time_to_next_batch = max(0, 20 - (time.time() - start))
            print(f"\twaiting {time_to_next_batch:02f}s for next batch...")
            time.sleep(time_to_next_batch)
        print("Done")

    elif args.runmode == "alvis":
        print("Running on Alvis")
        in_dir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/")
        model_path = in_dir / Path("models/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer_path = in_dir / Path("tokenizers/Mistral-7B-Instruct-v0.2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        distillator = ConversationalPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            framework="pt",
        )

        for i, prompt in enumerate(prompts):
            conversation = Conversation(prompt["text"])
            response = distillator(conversation)
            answer = response.messages[-1]["content"]
            with open(args.out_dir / Path(f"{str(i).zfill(5)}.txt"), "w") as f:
                f.write(answer)

    else:
        raise ValueError(f"Invalid runmode argument {runmode}")
