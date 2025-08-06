import locale
import os
from enum import Enum
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

try:
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
except locale.Error:
    print("⚠️ Locale 'fr_FR.UTF-8' non trouvée. Format par défaut utilisé.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "ERREUR CRITIQUE: La variable d'environnement GEMINI_API_KEY n'est pas définie. L'application ne peut pas démarrer.")
else:
    print("✅ Clé API GEMINI chargée avec succès.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com")
print(f"✅ Modèle Gemini configuré : {GEMINI_MODEL}")


@app.get("/")
async def root():
    return {"message": "Hello World"}


async def send_to_gemini(prompt: str) -> str:
    print("\n--- Envoi de la requête à l'API Gemini... ---")
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            print("--- ✅ Réponse reçue de Gemini. ---")
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = e.response.text
            print(f"[ERREUR HTTP {status_code}] Gemini a retourné une erreur : {error_detail}")
            if status_code == 429:
                raise HTTPException(status_code=503,
                                    detail="Le service d'analyse IA est surchargé. Veuillez réessayer plus tard.")
            elif status_code in [400, 401, 403]:
                raise HTTPException(status_code=500,
                                    detail="Problème de configuration interne avec le service d'analyse IA.")
            else:
                raise HTTPException(status_code=502, detail="Le service d'analyse IA externe rencontre un problème.")
        except httpx.RequestError as e:
            print(f"[ERREUR RÉSEAU] Impossible de contacter Gemini : {e}")
            raise HTTPException(status_code=504,
                                detail="Impossible de joindre le service d'analyse IA. Vérifiez votre connexion.")
        except (KeyError, IndexError):
            raise HTTPException(status_code=500,
                                detail="La réponse du service d'analyse IA est dans un format inattendu.")


class LanguageEnum(str, Enum):
    fr = "fr"
    en = "en"


class RecipeRequest(BaseModel):
    ingredients: List[str]
    language: LanguageEnum


@app.post("/generate-recipe")
async def generate_recipe(data: RecipeRequest):
    ingredients = ", ".join(data.ingredients)

    prompt_fr = f"""
    Tu es un chef cuisinier créatif et passionné. 
    Ton rôle est de créer une recette simple et délicieuse à partir des ingrédients suivants : {ingredients}.

    Ta réponse doit être au format JSON et contenir les clés suivantes : 
    - nom_recette
    - description_courte (2 phrases alléchantes)
    - instructions (une liste d'étapes claires)

    Utilise un style pédagogue et encourageant. Utilise du markdown.
    Réponds en français.
    N'hésite pas à ajouter des conseils ou des astuces pour réussir la recette.
    Utilise des émojis pour rendre la recette plus engageante.
    """.strip()

    prompt_en = f"""
    You're a creative and passionate chef. 
    Your role is to create a simple, delicious recipe using the following ingredients: {ingredients}.

    Your answer must be in JSON format and contain the following keys:
    - recipe_name
    - short_description (2 tantalizing sentences)
    - instructions (a list of clear steps)

    Use a pedagogical and encouraging style. Use markdown.
    Respond in English.
    Feel free to add tips or tricks to make the recipe a success.
    Use emojis to make the recipe more engaging.
    """.strip()

    print("--- ✅ Prompt intelligent construit. Prêt pour l'envoi. ---")

    prompt = prompt_fr if data.language == LanguageEnum.fr else prompt_en
    gemini_response = await send_to_gemini(prompt)

    print("\n🎉 Recette générée avec succès ! 🎉")

    return {
        "message": "Recette générée avec succès 🎉",
        "recipe": gemini_response
    }
