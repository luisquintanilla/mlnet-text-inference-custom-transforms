using Microsoft.ML;
using MLNet.TextInference.Onnx;

// Paths — download model from https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl (ONNX export)
var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var tokenizerPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models"));

Console.WriteLine("=== Multilingual NER (Davlan/bert-base-multilingual-cased-ner-hrl) ===\n");

var mlContext = new MLContext();

// BIO labels for Davlan/bert-base-multilingual-cased-ner-hrl
// string[] labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"];
string[] labels = ["O", "B-DATE", "I-DATE", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"];

/*
 https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/blob/main/onnx/config.json

 "id2label": {
    "0": "O",
    "1": "B-DATE",
    "2": "I-DATE",
    "3": "B-PER",
    "4": "I-PER",
    "5": "B-ORG",
    "6": "I-ORG",
    "7": "B-LOC",
    "8": "I-LOC"
  } 
 */


/*
bert-base-multilingual-cased-ner-hrl is a Named Entity Recognition model created by Davlan that identifies 
three entity types—person (PER), organization (ORG), and location (LOC)—across 10 languages: 
Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese.

Abbreviation	Description
O	    Outside of a named entity
B-PER	Beginning of a person’s name right after another person’s name
I-PER	Person’s name
B-ORG	Beginning of an organisation right after another organisation
I-ORG	Organisation
B-LOC	Beginning of a location right after another location
I-LOC	Location
 */

var nerOptions = new OnnxNerOptions
{
    ModelPath = modelPath,
    TokenizerPath = tokenizerPath,
    Labels = labels,
    InputColumnName = "Text",
    OutputColumnName = "Entities",
    MaxTokenLength = 128,
    BatchSize = 8
};

var sampleData = new[]
{
    new TextData { Text = "John Smith works at Microsoft in Seattle." },
    new TextData { Text = "Marie Curie a travaillé à l'Université de Paris." },
    new TextData { Text = "Angela Merkel war Bundeskanzlerin von Deutschland." },
    new TextData { Text = "東京タワーは日本の東京都港区にあります。" },
    new TextData { Text = "Elon Musk founded SpaceX and Tesla." },
    new TextData { Text = "Barack Obama was the 44th President of the United States." },
    new TextData { Text = "The Eiffel Tower is located in Paris, France." },
    new TextData { Text = "The Great Wall of China is a historic landmark." },
    new TextData { Text = "The Amazon rainforest is home to diverse wildlife." },
    new TextData { Text = "The Taj Mahal is a famous mausoleum in India." },
    new TextData { Text = "The Sydney Opera House is an iconic building in Australia." },
    new TextData { Text = "The Colosseum is an ancient amphitheater in Rome, Italy." },
    new TextData { Text = "The Statue of Liberty is a symbol of freedom in the United States." },
    new TextData { Text = "The Great Barrier Reef is the world's largest coral reef system." },
    new TextData { Text = "The Pyramids of Giza are ancient structures in Egypt." },
    new TextData { Text = "The Kremlin is a historic fortified complex in Moscow, Russia." },
    new TextData { Text = "The Louvre Museum is a famous art museum in Paris, France." },
    new TextData { Text = "The Golden Gate Bridge is a suspension bridge in San Francisco, California." },
    new TextData { Text = "The Acropolis is an ancient citadel in Athens, Greece." },
    new TextData { Text = "The Great Sphinx of Giza is a limestone statue in Egypt." },
    new TextData { Text = "The Burj Khalifa is the tallest building in the world, located in Dubai, United Arab Emirates." },
    new TextData { Text = "The Great Wall of China is a UNESCO World Heritage site." },
    new TextData { Text = "The Amazon River is the second longest river in the world." },
    new TextData { Text = "The Sahara Desert is the largest hot desert in the world." },
    new TextData { Text = "François Hollande a été président de la France." },
    new TextData { Text = "Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor, and investor. A pioneer of the personal computer revolution of the 1970s and 1980s, Jobs co-founded Apple Inc. with his early business partner Steve Wozniak as Apple Computer Company in 1976." },
    new TextData { Text = "Elon Musk (born June 28, 1971) is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX; early investor, CEO, and product architect of Tesla, Inc.; founder of The Boring Company; co-founder of Neuralink; and co-founder and initial co-chairman of OpenAI." },
    new TextData { Text = "Jobs was born in San Francisco in 1955 and adopted shortly afterward. Jobs co-founded Apple Inc. with his early business partner Steve Wozniak." },
    new TextData { Text = "Barack Hussein Obama II (born August 4, 1961) is an American former politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American to serve as president. Obama represented Illinois in the United States Senate from 2005 to 2008 and served as an Illinois state senator from 1997 to 2004." },
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

Console.WriteLine("Fitting multilingual NER pipeline...");
var estimator = mlContext.Transforms.OnnxNer(nerOptions);
var transformer = estimator.Fit(dataView);

var texts = sampleData.Select(s => s.Text).ToList();
var entities = transformer.ExtractEntities(texts);

for (int i = 0; i < texts.Count; i++)
{
    Console.WriteLine($"\nText: \"{texts[i]}\"");
    if (entities[i].Length == 0)
    {
        Console.WriteLine("  (no entities found)");
    }
    else
    {
        foreach (var e in entities[i])
        {
            Console.WriteLine($"  {e.EntityType}: \"{e.Word}\" [{e.StartChar}..{e.EndChar}] (score: {e.Score:F4})");
        }
    }
}

Console.WriteLine("\nDone!");
transformer.Dispose();

public class TextData
{
    public string Text { get; set; } = "";
}
