# Databricks notebook source
import json
import random

# Base texts for positive (tax-related) and negative (non-tax) samples
positive_texts = [
    "Rate 2% of gross revenues from UK in-scope activities above threshold; however, taxpayers may apply an alternative calculation method calculated based on operating margin in respect of in-scope activities where they are loss-making or have a very low profit margin Thresholds £500m revenues from in-scope activities provided globally and £25m of revenue from in-scope activities provided to UK users per 12-month accounting period.",
    "The Court formulated the following principle of law for the lower court to follow: in transfer pricing cases involving intra-group transfers of goods between two low-risk companies, where risk is reduced due to the existence of a single production centre essentially producing against confirmed orders, TNMM is more appropriate than CUP because profit margin is a more indicative criterion than nominal price. In such a model, nominal price is not the result of a free market. According to the Court, TNMM with an ROS indicator is fully acceptable and may be preferred to CUP in centrally controlled, low-risk distribution models where prices are not determined by free market forces.",
    "Belanghebbende is een in Nederland gevestigde vennootschap die behoort tot een internationaal opererend concern. Aan belanghebbende zijn voor de jaren 2008 tot en met 2016 (navorderings)aanslagen vennootschapsbelasting (vpb) opgelegd met aanzienlijke correcties op de door belanghebbende aangegeven belastbare bedragen. Bovendien heeft de inspecteur voor de jaren 2010 en 2012 tot en met 2016 vergrijpboeten opgelegd. Voor alle jaren is in geschil of de vergoedingen (verrekenprijzen) die diverse in het buitenland gevestigde concernvennootschappen voor leveringen en diensten aan belanghebbende in rekening hebben gebracht als zakelijk kunnen worden aangemerkt. Daarnaast is voor het jaar 2016 in geschil of de beëindiging van door een gevoegde dochtermaatschappij van belanghebbende geëxploiteerde licentierechten dient te worden aangemerkt als een onzakelijke onttrekking aan het vermogen van belanghebbende.",
    "Aveva, quindi, acclarato che la contribuente aveva acquistato i beni dalla capogruppo ad un prezzo lontano dal ‘valore normale’ di cui al citato art. 110, comma 7. La società impugnava, con tre distinti ricorsi, gli atti impositivi innanzi alla Commissione tributaria provinciale di Firenze, che, previa riunione, li accoglieva: annullava il rilievo relativo al evidenziando che già era stata adottata, per altra annualità, una pronuncia favorevole alla contribuente, sulla base della non applicabilità della metodologia del TNMM e che l’Ufficio aveva errato nella scelta dei ‘comparabili’; annullava il rilievo relativo all’IVA in"
]

negative_texts = [
    "One of the most urgent international climate issues today is the accelerating rise in global temperatures, driven primarily by human activity. This warming trend—commonly referred to as global warming—is the result of increased concentrations of greenhouse gases such as carbon dioxide, methane, and nitrous oxide in the atmosphere. These gases trap heat, causing the Earth’s surface and oceans to warm at unprecedented rates.",
    "In an increasingly interconnected world, health threats can cross borders within hours, challenging national systems and placing significant pressure on global health institutions. Whether emerging from infectious diseases, environmental hazards, geopolitical disruptions, or chronic conditions accelerated by modern lifestyles, these risks remind us that public health is a collective responsibility. This advisory outlines essential principles, recommended strategies, and coordinated actions for individuals, governments, and international organizations to safeguard public health and promote resilient health systems worldwide."
]

# Function to generate a long text (~1000 words) by repeating and trimming
def generate_long_text(base_text, target_words=1000):
    words = base_text.split()
    repeats = target_words // len(words) + 1
    long_text = ' '.join(words * repeats)
    return ' '.join(long_text.split()[:target_words])

# Generate dataset
num_positive = 50
num_negative = 50
dataset = []

for _ in range(num_positive):
    base = random.choice(positive_texts)
    long_text = generate_long_text(base, 1000)
    dataset.append({"doc_text": long_text, "true_label": 1})

for _ in range(num_negative):
    base = random.choice(negative_texts)
    long_text = generate_long_text(base, 1000)
    dataset.append({"doc_text": long_text, "true_label": 0})

# Shuffle dataset
random.shuffle(dataset)

# Save to JSON
with open("synthetic_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Generated {len(dataset)} samples.")
