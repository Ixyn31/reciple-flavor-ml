import re
import sys
import requests
import spacy
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pint import UnitRegistry

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config

# Load spaCy's small English pipeline model and Pint Unit Registry
nlp = spacy.load("en_core_web_sm")
ureg = UnitRegistry()

# --- Pydantic Data Schemas ---

class ParsedIngredient(BaseModel):
    raw_text: str
    base_ingredient: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    detected_techniques: List[str] = Field(default_factory=list)

class NutritionProfile(BaseModel):
    calories: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    carbs_g: float = 0.0

class EnrichedIngredient(BaseModel):
    parsed: ParsedIngredient
    nutrition_per_100g: NutritionProfile
    estimated_portion_grams: Optional[float] = None
    scaled_nutrition: Optional[NutritionProfile] = None


# --- Pipeline Logic ---

MEASUREMENT_UNITS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon",
    "teaspoons", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
    "g", "gram", "grams", "kg", "clove", "cloves", "pinch", "dash"
}

KNOWN_TECHNIQUES = {
    "roast", "saute", "boil", "fry", "grill", "char", "caramelize", 
    "bake", "toast", "mince", "chop", "dice", "slice", "shred", "sear"
}


class RecipePipeline:
    def __init__(self, usda_api_key: str = config.USDA_API_KEY):
        self.usda_api_key = usda_api_key

    def fetch_usda_nutrition(self, ingredient_query: str) -> tuple[NutritionProfile, dict]:
        """Fetches 100g nutrients along with USDA's built-in portion measures."""
        if not ingredient_query:
            return NutritionProfile(), {}

        endpoint = f"{config.USDA_BASE_URL}/foods/search"
        params = {
            "api_key": self.usda_api_key,
            "query": f"{ingredient_query} raw",
            "pageSize": 5,
            "dataType": ["Foundation", "SR Legacy"]
        }
        
        try:
            res = requests.get(endpoint, params=params, timeout=5)
            res.raise_for_status()
            data = res.json()
            foods = data.get("foods", [])
            
            # Fallback search if query + 'raw' returned 0 results
            if not foods:
                params["query"] = ingredient_query
                res = requests.get(endpoint, params=params, timeout=5)
                res.raise_for_status()
                foods = res.json().get("foods", [])

            if not foods:
                return NutritionProfile(), {}

            # Pick best food match
            selected_food = foods[0]
            for f in foods:
                desc = f.get("description", "").lower()
                if "raw" in desc and ingredient_query.lower() in desc:
                    selected_food = f
                    break

            # 1. Robust Case-Insensitive Nutrient Parsing (Per 100g)
            nutrition = NutritionProfile()
            food_nutrients = selected_food.get("foodNutrients", [])

            for nutrient in food_nutrients:
                name = nutrient.get("nutrientName", "").lower()
                unit_name = str(nutrient.get("unitName", "")).upper()
                val = float(nutrient.get("value", 0.0))

                if "energy" in name and (unit_name == "KCAL" or "kcal" in name):
                    nutrition.calories = val
                elif "protein" in name:
                    nutrition.protein_g = val
                elif "total lipid" in name or "fat" in name:
                    nutrition.fat_g = val
                elif "carbohydrate" in name:
                    nutrition.carbs_g = val

            # 2. Extract Dynamic Portion Weights
            portion_map = {}
            for portion in selected_food.get("foodPortions", []):
                modifier = portion.get("modifier", "") or portion.get("portionDescription", "")
                modifier = modifier.lower().strip()
                gram_wt = float(portion.get("gramWeight", 0.0))
                if modifier and gram_wt > 0:
                    portion_map[modifier] = gram_wt

            return nutrition, portion_map

        except requests.RequestException as e:
            print(f"⚠️ USDA API Error for '{ingredient_query}': {e}")
            return NutritionProfile(), {}
        except Exception as e:
            print(f"⚠️ Data Parsing Error for '{ingredient_query}': {e}")
            return NutritionProfile(), {}

    def parse_ingredient_text(self, raw_ingredient: str) -> ParsedIngredient:
        """Parses a raw ingredient string into structured entities."""
        text_lower = raw_ingredient.lower().strip()
        
        # 1. Parse Quantity
        quantity = None
        qty_match = re.search(r"^(\d+\/\d+|\d+\.\d+|\d+)", text_lower)
        if qty_match:
            val = qty_match.group(1)
            if "/" in val:
                num, denom = val.split("/")
                quantity = round(float(num) / float(denom), 3)
            else:
                quantity = float(val)

        # 2. Process with spaCy NLP Pipeline
        doc = nlp(text_lower)
        
        unit = None
        detected_techniques = []
        base_tokens = []

        for token in doc:
            lemma = token.lemma_
            text = token.text
            
            # Extract unit
            if text in MEASUREMENT_UNITS or lemma in MEASUREMENT_UNITS:
                if not unit:
                    unit = text
                continue

            # Extract technique verbs/adjectives
            if lemma in KNOWN_TECHNIQUES or text in KNOWN_TECHNIQUES:
                detected_techniques.append(lemma)
                continue

            # Exclude additional common prep adjectives/verbs from the base entity string
            if text in ["roasted", "minced", "chopped", "diced", "sliced", "caramelized", "extra-virgin", "virgin"]:
                continue

            # Retain core Nouns/Adjectives for base food entity
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop:
                if not token.like_num and text not in ["-"]:
                    base_tokens.append(text)

        base_ingredient = " ".join(base_tokens).strip()

        return ParsedIngredient(
            raw_text=raw_ingredient,
            base_ingredient=base_ingredient if base_ingredient else text_lower,
            quantity=quantity,
            unit=unit,
            detected_techniques=list(set(detected_techniques))
        )

    def get_estimated_grams(self, quantity: Optional[float], unit: Optional[str], portion_map: dict) -> float:
        """Determines gram weight dynamically using USDA portions or Pint unit conversions."""
        qty = quantity if quantity is not None else 1.0
        
        if not unit:
            return 100.0
            
        unit_clean = unit.lower().strip()

        # 1. Check USDA Dynamic Portions
        for modifier, gram_wt in portion_map.items():
            if unit_clean in modifier or modifier in unit_clean:
                return round(qty * gram_wt, 2)

        # 2. Pint Unit Registry
        try:
            quantity_obj = qty * ureg(unit_clean)
            if quantity_obj.check('[mass]'):
                return round(float(quantity_obj.to(ureg.gram).magnitude), 2)
            elif quantity_obj.check('[volume]'):
                return round(float(quantity_obj.to(ureg.milliliter).magnitude), 2)
        except Exception:
            pass

        return 100.0

    def process_ingredient(self, raw_ingredient: str) -> EnrichedIngredient:
        parsed = self.parse_ingredient_text(raw_ingredient)
        
        # Unpack USDA nutrition and portion measures
        nutrition_100g, portion_map = self.fetch_usda_nutrition(parsed.base_ingredient)
        
        # Calculate portion grams rounded to 2 decimal places
        total_grams = self.get_estimated_grams(parsed.quantity, parsed.unit, portion_map)
        
        # Scale nutrition values
        scale_factor = total_grams / 100.0 if total_grams else 1.0
        scaled_nutrition = NutritionProfile(
            calories=round(nutrition_100g.calories * scale_factor, 2),
            protein_g=round(nutrition_100g.protein_g * scale_factor, 2),
            fat_g=round(nutrition_100g.fat_g * scale_factor, 2),
            carbs_g=round(nutrition_100g.carbs_g * scale_factor, 2),
        )
        
        return EnrichedIngredient(
            parsed=parsed,
            nutrition_per_100g=nutrition_100g,
            estimated_portion_grams=total_grams,
            scaled_nutrition=scaled_nutrition
        )


if __name__ == "__main__":
    pipeline = RecipePipeline()
    
    samples = [
        "3 cloves roasted garlic, minced",
        "1/2 cup caramelized yellow onions",
        "2 tbsp extra-virgin olive oil"
    ]
    
    print("\n=== PIPELINE RUN DEMO ===\n")
    for raw in samples:
        result = pipeline.process_ingredient(raw)
        print(f"Raw Input         : {result.parsed.raw_text}")
        print(f"Base Entity       : {result.parsed.base_ingredient}")
        print(f"Quantity          : {result.parsed.quantity} {result.parsed.unit}")
        print(f"Techniques        : {result.parsed.detected_techniques}")
        print(f"Portion Grams     : {result.estimated_portion_grams}g")
        print(f"Scaled Nutrition  : {result.scaled_nutrition.model_dump()}")
        print("-" * 50)