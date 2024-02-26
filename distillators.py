from runner import LLMRunner, RunMode


class Distillator(LLMRunner):
    def __init__(self, runmode: RunMode):
        super().__init__(runmode)

    def _make_prompt(self, text: str) -> str:
        assert False, "not implemented"

    def distill(self, text: str) -> (str, str):
        prompt = self._make_prompt(text)
        response = self.run(prompt)
        return prompt, response


class AmazonReviewDistillator(Distillator):
    def __init__(self, runmode: RunMode):
        super().__init__(runmode)

    def _make_prompt(self, text: str) -> str:
        return f"""
            Rewrite the review such that the sentiment is completely neutral. It is
            very important that one cannot tell whether the review is positive or
            negative at all. Try and keep all other information in the review.
            
            Here's the review:

            {text}
            """


class SyntheticDataDistillator(Distillator):
    def __init__(self, runmode: RunMode):
        super().__init__(runmode)

    def _make_prompt(self, text: str) -> str:
        return f"""
            Rewrite the text such that any information about whether the country
            joined an IMF program or not is completely removed, while keeping any
            other information about the country intact.
            
            Here's the text:

            {text}
            """


class SyntheticDataDistillatorDGP(Distillator):
    def __init__(self, runmode: RunMode):
        super().__init__(runmode)

    def _make_prompt(self, text: str) -> str:
        return f"""
            The following text was created by an GPT 2 model. Each paragraph was
            generated independently from a prompt that either:

            1). asked the model to write a generic paragraph about the country

                Example: "Antigua and Barbuda."

            2). asked the model to write a paragraph about how the country asked for
              an IMF program 

                Example: "Antigua and Barbuda's government has asked the IMF for a program."

            3). asked the model to write a paragraph about the demands of the IMF on
            the country

                Example: "International Monetary Fund: No labor policy liberalization in Antigua and Barbuda."
           
            Given how the data was generated, remove paragraphs that were generated
            from a prompt of type 3 completely, while keeping the other paragraphs
            exactly the same.

            Here's the text:

            {text}
            """
