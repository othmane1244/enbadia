"""
create_graphs.py — Génère les 5 graphiques matplotlib
"""

import json
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(metrics_file: str) -> dict:
    """Charge les métriques."""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_graphs(metrics_data, output_dir="reports"):
    """Crée les 5 graphiques."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    stats = metrics_data.get("statistics", {})
    frames = metrics_data.get("frames", [])
    alerts = metrics_data.get("alerts", [])
    
    # Extraire les latences
    latencies = [f.get('latency_ms', 0) for f in frames if 'latency_ms' in f]
    
    print(f"📊 Création des graphiques...")
    print(f"   - {len(frames)} frames collectées")
    print(f"   - {len(alerts)} alertes générées")
    print(f"   - Latence: min={min(latencies):.2f}ms, max={max(latencies):.2f}ms")
    
    # 1. Latence dans le temps
    plt.figure(figsize=(14, 6))
    frame_ids = list(range(len(latencies)))
    plt.plot(frame_ids, latencies, color='#667eea', linewidth=1.5, alpha=0.7, label='Latence')
    plt.axhline(y=np.mean(latencies), color='#e63946', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(latencies):.2f}ms')
    plt.xlabel('Frame ID', fontsize=12, fontweight='bold')
    plt.ylabel('Latence (ms)', fontsize=12, fontweight='bold')
    plt.title('Latence Frame → Alerte dans le Temps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "01_latence_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 01_latence_timeline.png")
    
    # 2. Distribution latence
    plt.figure(figsize=(12, 6))
    plt.hist(latencies, bins=50, color='#667eea', alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(latencies), color='#e63946', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(latencies):.2f}ms')
    plt.xlabel('Latence (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Fréquence', fontsize=12, fontweight='bold')
    plt.title('Distribution de la Latence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "02_latence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 02_latence_distribution.png")
    
    # 3. Alertes par type (camembert)
    alerts_by_type = stats.get('alerts_by_type', {})
    if alerts_by_type:
        plt.figure(figsize=(10, 8))
        colors = ['#E63946', '#F77F00', '#06A77D', '#4A90E2']
        types = list(alerts_by_type.keys())
        counts = list(alerts_by_type.values())
        
        wedges, texts, autotexts = plt.pie(counts, labels=types, autopct='%1.1f%%', 
                                            colors=colors[:len(types)], startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.title('Répartition des Alertes par Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "03_alertes_par_type.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ 03_alertes_par_type.png")
    
    # 4. FPS et alertes
    fps = stats.get('fps_average', 0)
    total_frames = stats.get('frames_processed', 0)
    total_alerts = stats.get('alerts_generated', 0)
    alert_ratio = (total_alerts / total_frames * 100) if total_frames > 0 else 0
    
    plt.figure(figsize=(12, 6))
    x = np.arange(2)
    values = [fps, alert_ratio]
    colors_bar = ['#06A77D', '#F77F00']
    bars = plt.bar(x, values, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    plt.xticks(x, ['FPS Moyen', 'Ratio Alertes (%)'])
    plt.ylabel('Valeur', fontsize=12, fontweight='bold')
    plt.title('Performance: FPS et Ratio Alertes', fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs sur les barres
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "04_fps_et_alertes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 04_fps_et_alertes.png")
    
    # 5. Boîte à moustaches latence
    plt.figure(figsize=(10, 6))
    box_data = [latencies]
    bp = plt.boxplot(box_data, labels=['Latence'], patch_artist=True, widths=0.5)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#667eea')
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set_color('#333')
        whisker.set_linewidth(1.5)
    
    for cap in bp['caps']:
        cap.set_color('#333')
        cap.set_linewidth(1.5)
    
    plt.ylabel('Latence (ms)', fontsize=12, fontweight='bold')
    plt.title('Analyse de la Latence (Boîte à Moustaches)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    
    # Ajouter les stats
    q1 = np.percentile(latencies, 25)
    q2 = np.percentile(latencies, 50)
    q3 = np.percentile(latencies, 75)
    
    stats_text = f'Q1: {q1:.2f}ms | Médiane: {q2:.2f}ms | Q3: {q3:.2f}ms'
    plt.text(0.5, 0.02, stats_text, ha='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "05_latence_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 05_latence_boxplot.png")
    
    print(f"\n✅ Tous les graphiques ont été créés dans {output_dir}/")


def main():
    """Génère les graphiques."""
    metrics_file = "reports/metrics.json"
    
    if not Path(metrics_file).exists():
        print(f"❌ Fichier de métriques non trouvé : {metrics_file}")
        return False
    
    print(f"📖 Chargement des métriques...")
    metrics_data = load_metrics(metrics_file)
    
    create_graphs(metrics_data)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
