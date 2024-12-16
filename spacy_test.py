import spacy

model = spacy.load("en_core_web_sm")
model.to_disk("./en_core_web_sm")
