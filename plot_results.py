import os
import csv
import math

def parse_csv(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def generate_svg_pr_curve(pr_data, output_path):
    # Separate by model
    models = {}
    for row in pr_data:
        m = row['model']
        if m not in models:
            models[m] = []
        try:
            r = float(row['recall'])
            p = float(row['precision'])
            models[m].append((r, p))
        except ValueError:
            continue

    width, height = 600, 450
    padding = 60
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    colors = {'LSG': '#2b5c8f', 'Baseline': '#d95f02'}

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background-color:#ffffff; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Precision-Recall Curve</text>')
    
    # Axes
    x0, y0 = padding, height - padding
    x1, y1 = width - padding, padding
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    # Grid & Labels
    for i in range(6):
        val = i / 5.0
        cx = x0 + val * plot_w
        cy = y0 - val * plot_h
        svg.append(f'<line x1="{cx}" y1="{y0}" x2="{cx}" y2="{y1}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{cx}" y="{y0 + 20}" text-anchor="middle" font-size="12">{val:.1f}</text>')
        svg.append(f'<text x="{x0 - 10}" y="{cy + 4}" text-anchor="end" font-size="12">{val:.1f}</text>')

    svg.append(f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="13">Recall</text>')
    svg.append(f'<text x="15" y="{height/2}" text-anchor="middle" font-size="13" transform="rotate(-90 15 {height/2})">Precision</text>')

    # Plots
    color_idx = 0
    palette = ['#2b5c8f', '#d95f02', '#2ca02c', '#9467bd']
    legend_y = y1 + 20

    for m_name, points in models.items():
        points.sort(key=lambda item: item[0])
        c = colors.get(m_name, palette[color_idx % len(palette)])
        color_idx += 1
        
        path_data = []
        for r, p in points:
            cx = x0 + r * plot_w
            cy = y0 - p * plot_h
            path_data.append(f"{cx:.1f},{cy:.1f}")
        
        svg.append(f'<polyline points="{" ".join(path_data)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        svg.append(f'<rect x="{x1 - 140}" y="{legend_y}" width="12" height="12" fill="{c}"/>')
        svg.append(f'<text x="{x1 - 120}" y="{legend_y + 10}" font-size="12">{m_name}</text>')
        legend_y += 20

    svg.append('</svg>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))

def generate_svg_latency(lat_data, output_path):
    if not lat_data:
        return
    
    width, height = 600, 450
    padding = 60
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    x_vals = [float(r['memory_size']) for r in lat_data]
    lsg_vals = [float(r['lsg_latency_ms']) for r in lat_data]
    base_vals = [float(r['baseline_latency_ms']) for r in lat_data]

    max_x = max(x_vals)
    max_y = max(base_vals) * 1.1

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background-color:#ffffff; font-family:sans-serif;">']
    svg.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="16" font-weight="bold">Inference Latency vs. Memory Size</text>')

    x0, y0 = padding, height - padding
    x1, y1 = width - padding, padding
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    # Grid
    for i in range(5):
        val_y = (max_y / 4.0) * i
        cy = y0 - (val_y / max_y) * plot_h
        svg.append(f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 10}" y="{cy + 4}" text-anchor="end" font-size="12">{val_y:.1f}</text>')

    svg.append(f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="13">Memory Store Size (n items)</text>')
    svg.append(f'<text x="15" y="{height/2}" text-anchor="middle" font-size="13" transform="rotate(-90 15 {height/2})">Latency (ms)</text>')

    lsg_points = []
    base_points = []
    for x, y_l, y_b in zip(x_vals, lsg_vals, base_vals):
        cx = x0 + (x / max_x) * plot_w
        cy_l = y0 - (y_l / max_y) * plot_h
        cy_b = y0 - (y_b / max_y) * plot_h
        lsg_points.append(f"{cx:.1f},{cy_l:.1f}")
        base_points.append(f"{cx:.1f},{cy_b:.1f}")
        svg.append(f'<text x="{cx}" y="{y0 + 20}" text-anchor="middle" font-size="11">{int(x)}</text>')

    svg.append(f'<polyline points="{" ".join(lsg_points)}" fill="none" stroke="#2b5c8f" stroke-width="3"/>')
    svg.append(f'<polyline points="{" ".join(base_points)}" fill="none" stroke="#d95f02" stroke-width="3" stroke-dasharray="5,5"/>')

    # Legend
    svg.append(f'<rect x="{x0 + 20}" y="{y1 + 10}" width="12" height="12" fill="#2b5c8f"/>')
    svg.append(f'<text x="{x0 + 40}" y="{y1 + 20}" font-size="12">LSG O(1)</text>')
    svg.append(f'<rect x="{x0 + 120}" y="{y1 + 10}" width="12" height="12" fill="#d95f02"/>')
    svg.append(f'<text x="{x0 + 140}" y="{y1 + 20}" font-size="12">Baseline O(n)</text>')

    svg.append('</svg>')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))

def plot_all_results(results_dir="results2", assets_dir="assets"):
    os.makedirs(assets_dir, exist_ok=True)
    print(f"Generating visual graph assets from '{results_dir}' into '{assets_dir}'...")

    pr_data = parse_csv(os.path.join(results_dir, "pr_sweep.csv"))
    if pr_data:
        generate_svg_pr_curve(pr_data, os.path.join(assets_dir, "fig_pr_curve.svg"))
        print(" -> Generated assets/fig_pr_curve.svg")

    lat_data = parse_csv(os.path.join(results_dir, "latency_scaling.csv"))
    if lat_data:
        generate_svg_latency(lat_data, os.path.join(assets_dir, "fig_latency_scaling.svg"))
        print(" -> Generated assets/fig_latency_scaling.svg")

if __name__ == "__main__":
    plot_all_results()
