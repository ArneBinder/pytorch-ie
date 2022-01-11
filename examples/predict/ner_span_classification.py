from pytorch_ie import Document, Pipeline
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


model_name_or_path = "pie/example-ner-spanclf-conll03"
ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

document = Document(
    "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
)

ner_pipeline(document, predict_field="entities")

for entity in document.predictions("entities"):
    entity_text = document.text[entity.start : entity.end]
    label = entity.label
    print(f"{entity_text} -> {label}")
