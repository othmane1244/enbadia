from pathlib import Path
import os

print("🔍 Recherche de votre modèle...")
print("-" * 50)

# Dossiers à chercher
folders = [
    r"C:\Users\user\Documents\projet\fine-tuning 1",
    r"C:\Users\user\Documents\projet\fine_tuning2",
    r".",
    r".."
]

found = []
for folder in folders:
    if os.path.exists(folder):
        path = Path(folder)
        for pt_file in path.rglob("*.pt"):
            size_mb = pt_file.stat().st_size / (1024*1024)
            found.append((str(pt_file), size_mb))
            print(f"✅ Trouvé: {pt_file}")
            print(f"   Taille: {size_mb:.1f} MB")
            print()

if not found:
    print("❌ Aucun modèle .pt trouvé!")
    print("Vérifiez que l'entraînement s'est bien terminé.")
else:
    print(f"\n📊 Total: {len(found)} modèle(s) trouvé(s)")
    print("Copiez le chemin ci-dessus pour votre script.")