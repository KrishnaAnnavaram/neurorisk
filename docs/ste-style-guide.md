# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of neurorisk and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the neurorisk documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **record** | One row of the table: the data of one patient. | sample, instance, observation |
| **label** | The `Diagnosis` column (0 or 1). | target, outcome, class (for the column) |
| **feature** | One input column that the schema permits for a model. | variable, predictor, attribute |
| **feature set** | A named whitelist of features (`A`, `B` or `C`) in `configs/feature_sets.json`. | feature group, configuration |
| **domain** | A named group of features inside a feature set, for example `cardiovascular`. | category, block |
| **ceiling set** | Feature set `C`. It contains cognitive tests and symptoms, so its score is an upper limit, not a risk-factor result. | full model, best model |
| **schema** | The column contract in `schema.py`: names, types, ranges and permitted values. | format, spec |
| **outer fold** | One part of the outer split. The outer folds measure the model. | test split, holdout |
| **inner fold** | One part of the split inside an outer training part. The inner folds tune and calibrate. | validation split |
| **nested CV** | Outer folds for measurement with inner folds for tuning and calibration. | cross-validation (alone) |
| **out-of-fold prediction** | The prediction for a record from the model that did not see that record. | test prediction |
| **risk** | The calibrated probability of the label for one record. | score, risk score, index |
| **risk band** | A label (`low`, `moderate`, `elevated`, `high`) for a range of risk. | category, level, tier |
| **observed rate** | The fraction of records with label 1 inside a group of records. | actual risk, true rate |
| **importance** | The mean drop of AUROC on held-out records when the values of a feature or a domain are shuffled. | weight, contribution |
| **baseline** | The `prior` model. It gives the training prevalence to every record. | dummy |
| **model card** | The Markdown file that gives the intended use, data, inputs, performance and limits of one model. | datasheet |
| **synthetic data** | Records that a generator made. They do not come from real patients. | fake data, sample data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Compare a table with the schema and report errors and warnings. |
| **select** | Copy only the columns of a feature set from a table. |
| **tune** | Choose hyperparameters with inner folds. |
| **calibrate** | Fit a map from model output to probability with inner folds. |
| **evaluate** | Run nested CV for one feature set and one model. |
| **compare** | Run nested CV for many feature sets and test the AUROC differences. |
| **train** | Fit one final model on all records and save it. |
| **predict** | Give the risk and the risk band for new records with a saved model. |
| **shuffle** | Change the order of the values of a column (or a domain) in random sequence. |
