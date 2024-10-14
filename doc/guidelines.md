# Prompt-writing guidelines

Please follow these general guidelines when writing prompts.

- Start the prompt by clearly mentioning what kind of input text will be provided to the model (data domain, abstracts vs full-texts, etc).
- Use new lines to separate various sections of the instruction. Common sections include:
  - The task definition (**all tasks**).
  - The output format (for structured outputs).
  - The input context -- for instance, a paragraph from a paper (**all tasks**).
  - A question or claim based on the input context.
- Clearly demarcate the start of the input text, using markers such as "Input: ", "Abstract: ", etc.
- For tasks with a finite number of answers (like text classification) or a fixed schema (like information extraction), the prompt should provide a complete list of all valid answer types expected in the output -- e.g. all answer choices or entity types.
- For structured tasks, the output format expected from the model should be clearly described. While describing the output format, providing toy examples is encouraged. Examples are included below.
- Don't add unecessary text to the output for instance, for a summary, the output should be just the summary, rather than `Summary: {{ summary }}`. This removes the output parsing required to perform evaluation.

## Guidelines by task type

Below are some guideline prompts for common task types. Variations in phrasing are fine, as is adjusting the prompt for the specifics of each task (e.g. whether the input context is an abstract or a full paper).

- [Prompt-writing guidelines](#prompt-writing-guidelines)
  - [Guidelines by task type](#guidelines-by-task-type)
    - [Question answering](#question-answering)
      - [Unattributed QA](#unattributed-qa)
      - [Attributed QA](#attributed-qa)
    - [Summarization](#summarization)
      - [Sample Template 1](#sample-template-1)
      - [Sample Template 2](#sample-template-2)
    - [Entailment / fact checking](#entailment--fact-checking)
    - [Text classification](#text-classification)
  - [Information extraction](#information-extraction)
    - [NER](#ner)
    - [Relation extraction](#relation-extraction)
    - [Event extraction](#event-extraction)
  - [Guidelines for Delimiter](#guidelines-for-delimiter)

### Question answering

Question answering has two main variants: In attributed QA, the model needs to provide evidence to justify its answer. For unattributed QA, it can just output the answer.

#### Unattributed QA

```jinja
<!-- The task definition -->
You will be shown an abstract from a scientific research paper, followed by a question about the abstract. Please answer the question with a single sentence. Do not include any text in your repsonse other than the answer.

<!-- The context -->
Abstract: {{abstract}}

|||

{answer}
```

Minor formatting variations, like adding a newline before the abstract, are fine:

```jinja
Abstract:
{{abstract}}
```

#### Attributed QA

```jinja
<!-- The task definition. -->
You will be shown sections from a scientific research paper, followed by a question about the paper. Please answer the question based on the contents of the paper.

<!-- Output formatting and example. -->
Your response should be a `json` object with two fields:
"answer": A succinct answer to the question, in your own words.
"evidence": An array of strings. Each should be an excerpt from the paper. Together, the evidence should serve as a justification for the answer.

For instance, for the question "What baselines did the authors compare against?", a sample response might be:

{
  "answer": "BERT and RoBERTa".
  "evidence": ["In our experiments, we compare the performance of our model against BERT.", "In additional experiments, we compare against RoBERTa."]
}

Do not include any text in your response other than the json. If the question is unanswerable given the provided excerpts, respond with the single word "null".

<!-- The context. -->
Paper: {{paper}}

<!-- Question based on the context. -->
Question: {{question}}

|||

{{answer}}
```

### Summarization

- **Input Description**: Mention what kind of input text will be provided to the model (e.g., data domain, abstracts vs full-texts, etc.). The model could be presented with multiple paragraphs, an abstract, specific sections, or the full text from a research paper in a specified domain or from a well-known venue.

- **Output Description**: The prompt should guide the model to generate a summary of the provided text. Specify the level of detail required in the summary, and if needed, provide additional task-specific instructions such as "covering the main claims or findings of the paper."

#### Sample Template 1

```jinja
<!-- The task definition -->
You will be presented with the abstract, introduction, and conclusion from a research paper. Please summarize the main contribution of the paper in a single sentence. Your response should include the summary and no additional text.

<!-- The context -->
paper: {{paper}}

|||

{{summary}}
```

#### Sample Template 2

```jinja
<!-- The task definition -->
You will be presented with {multiple paragraphs/an abstract/xxx sections/full-text} from a {<domain>/<well-known venue>/...} research paper. Given the {paragraphs/abstract/sections/full-text}, your task is to generate a {<level-of-details> e.g. concise} summary {Optional: <task-specific> e.g. covering main claims or findings}. Your response should include the summary and no additional text.

<!-- The context -->
<input_type>: {{ input }}

|||

{{ output }}
```

### Entailment / fact checking

```jinja
<!-- Task description. -->
You will be shown a scientific claim, and the abstract of a biomedical research paper. Each sentence from the abstract will be on a separate line. Your task is to return a JSON object with two fields:

<!-- Output formatting. -->
- "verdict": The fact-checking verdict. If the information in the abstract supports the claim, write "SUPPORT". If the abstract contradicts the claim, write "CONTRADICT". If the abstract does not provide enough information to arrive at a verdict, write "NEI" (for "not enough information").
- "evidence": An array of sentences providing evidence for the verdict. Please copy all relevant sentences verbatim from the abstract. If the verdict was "NEI", then return an empty array.

For instance, if the model were given the claim "smoking causes cancer", the output might be
{
  "verdict": "SUPPORT",
  "evidence": ["The results of our meta-analysis provide overwhelming support that cigarette smoking is a risk cause for lung cancer."]
}

<!-- Claim (similar to a question). -->
Claim: {{ claim }}

<!-- Context. -->
Abstract:
{{ abstract_with_newlines }}

|||

{{ output_json }}
```

### Text classification

```jinja
<!-- Task definition and answer choices. -->
Below is a citation sentence occurring in a scientific research paper. Please classify the citation intent of this sentence as one of the following:
- method: Cites the paper for its methodology or procedure.
- background: Cites the paper to provide background information.
- result: Cites the paper for its findings or results.

<!-- Output formatting. -->
Your answer should be a single word from the following list of options: ["method", "background", "result"]. Do not include any other text in your response.

<!-- Context -->
Citation sentence:
{{ string }}

|||

{{ answer_choices[label] }}
```

## Information extraction

In addition to the general guidelines shown at the top, please follow the following instructions when creating IE templates.

- The prompt should provide a complete list of all types expected in the output (entity types, relation types, etc.)
- Output format expected from the model should be clearly described. For NER tasks, we expect the output to be a json object in which entity types are keys, with values being lists of extracted entities belonging to the corresponding type. For RE tasks, we expect the output to be a list of tuples in which every tuple corresponds to a relation and follows the format [ent1, ent2, ..., entn, rel_type].
- While describing the output format, providing toy examples is encouraged. For example, the template for ChemProt provides the following example: {"Chemical" : ["Dexamethasone", ...], "Protein" : ["BRCA-1", ...]}.

### NER

```jinja
<!-- Task definition -->
You will be shown {an abstract/the full-text} from a {biomedical/computer science/...} research paper. Given this {abstract/full-text}, your task is to extract all unique entities of the following types: {ner_type_list}.

<!-- Output formatting -->
Please return the output as a JSON object of the format: {"type1" : ["example_entity", ...], "type2" : ["example_entity", ...]}. The keys should be entity types and values should be lists of extracted entities belonging to the corresponding type. Entity types with no matching entities should be assigned an empty array [].

Only output the JSON object and do not include any additional text.

<!-- Context -->
Abstract:

{{ abstract }}

|||

{{ ner_dict | tojson }}
```

### Relation extraction

```jinja
<!-- Task definition -->
Below is an abstract from a computer science research paper. Given this abstract, your task is to extract all unique relationships between entities of the following types:

- "USED-FOR": For instance, "Our method models user proficiency" -> ["method", "user proficiency", "USED-FOR"].
- "FEATURE-OF": For instance, "prior knowledge of the model" -> ["prior knowledge", "model", "FEATURE-OF"].
...

<!-- Output formatting and example. -->
Format your output as a json array. Each entry in the array should express a single relation, formatted as [<Entity_A>, <Entity_B>, <RELATION_A_B>]. An example output might look like:

[["neural networks", "classification", "USED-FOR"], ["neuron", "neural network", "PART-OF"]].

If no relations are found, please return an empty array [].

<!-- Context -->
Abstract:

{{ org_text}}

|||

{{ relations }}
```

### Event extraction

TODO. For now, see [annotated_materials_syntheses_events.yaml](../tasks/templates/annotated_materials_syntheses_events.yaml).


## Guidelines for Delimiter

- Use a newline to separate distinct parts of the input.
- Use `:` for the start of any major text chunk.
- Use two newlines to separate input components and output components. Make sure to check extra newlines.
- Notes:
    Jinja expressions might automatically insert '\n'. To manage excessive newlines, use Jinja's whitespace control feature by adding a minus sign (-) inside the braces at the start or end of a block.

    For example:

    ```plaintext
    Version 1:
    Answer Choices:
    {% for option in options %}
    {{ option.key }}: {{ option.value }}
    {% endfor %}

    # This will render
    A. A text

    B. B text

    C. C text
    ....

    Version 2:
    Answer Choices:
    {%- for option in options %}
    {{ option.key }}: {{ option.value }}{%- if not loop.last %}

    {%- endif %}
    {%- endfor %}

    # This will render
    A. A text
    B. B text
    C. C text
    ....
