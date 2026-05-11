"""
run_full_test.py — Lancer le simulateur sur N frames et générer rapports
Système de Surveillance Intelligente — ENSA Béni Mellal
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    num_frames = 2000  # Nombre de frames cible
    
    print("\n" + "="*70)
    print("🚀 LANCEMENT DU TEST COMPLET (Phase 5)")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Nombre frames cible: {num_frames}+")
    print(f"  - FPS simulé: ~30 fps")
    print(f"  - Durée estimée: ~{num_frames/30:.0f} secondes (~{num_frames/30/60:.1f} minutes)")
    print("="*70 + "\n")
    
    # Vérifier que reports/ existe
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Étape 1 : Vérifier que l'API est lancée
    print("📡 Vérification de l'API...")
    try:
        import httpx
        client = httpx.Client(timeout=2.0)
        try:
            resp = client.get("http://127.0.0.1:8000/docs")
            if resp.status_code == 200:
                print("✅ API est accessible sur http://127.0.0.1:8000")
            else:
                print("⚠️  API répond mais status != 200")
        except:
            print("❌ API non accessible sur http://127.0.0.1:8000")
            print("   Assurez-vous que l'API est lancée :")
            print("   >>> python -m uvicorn main:app --host 127.0.0.1 --port 8000")
            return False
    except ImportError:
        print("⚠️  httpx non disponible, vérification skipped")
    
    # Étape 2 : Lancer le simulateur
    print("\n📹 Lancement du simulateur...")
    print("-" * 70)
    
    try:
        # Lancer le simulateur en mode synchrone
        simulator_script = project_root / "model" / "simulator.py"
        result = subprocess.run(
            [sys.executable, str(simulator_script)],
            cwd=str(project_root),
            capture_output=False,
            timeout=None  # Pas de timeout — contrôlé manuellement via Ctrl+C
        )
        
        if result.returncode != 0:
            print(f"\n❌ Simulateur s'est arrêté avec le code {result.returncode}")
            return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulateur interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors du lancement: {e}")
        return False
    
    print("-" * 70)
    
    # Étape 3 : Générer les rapports
    print("\n📊 Génération des rapports et graphiques...")
    
    try:
        from generate_report import load_metrics, generate_performance_graphs, generate_html_report
        
        metrics_file = reports_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"❌ Fichier de métriques non trouvé: {metrics_file}")
            print("   Les métriques auraient dû être générées par le simulateur.")
            return False
        
        # Charger et afficher les métriques
        metrics_data = load_metrics(str(metrics_file))
        stats = metrics_data.get("statistics", {})
        
        print(f"\n📈 Statistiques collectées:")
        print(f"  - Frames traitées: {stats.get('frames_processed', 0)}")
        print(f"  - Alertes générées: {stats.get('alerts_generated', 0)}")
        print(f"  - FPS moyen: {stats.get('fps_average', 0):.2f}")
        print(f"  - Latence moyenne: {stats.get('latency_ms', {}).get('avg', 0):.2f} ms")
        
        # Générer les graphiques
        print(f"\n🎨 Génération des graphiques...")
        generate_performance_graphs(metrics_data, str(reports_dir))
        
        # Générer le rapport HTML
        print(f"\n📄 Génération du rapport HTML...")
        generate_html_report(metrics_data, str(reports_dir))
        
        # Afficher le résumé final
        from metrics_collector import MetricsCollector
        collector = MetricsCollector()
        collector.frames = metrics_data.get("frames", [])
        collector.alerts = metrics_data.get("alerts", [])
        collector.frame_count = len(collector.frames)
        collector.alert_count = len(collector.alerts)
        collector.start_time = 0
        collector.end_time = stats.get("total_time_seconds", 0)
        collector.print_summary()
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des rapports: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Étape 4 : Afficher les résultats
    print("\n" + "="*70)
    print("✅ TEST COMPLET TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print(f"\n📁 Fichiers générés dans: {reports_dir}/")
    print("  - metrics.json       (données brutes)")
    print("  - rapport_pfe.html   (rapport interactif)")
    print("  - 01_latence_timeline.png")
    print("  - 02_latence_distribution.png")
    print("  - 03_alertes_par_type.png")
    print("  - 04_fps_et_alertes.png")
    print("  - 05_latence_boxplot.png")
    print("\n🌐 Ouvrir le rapport HTML:")
    print(f"   {reports_dir / 'rapport_pfe.html'}")
    print("\n" + "="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
