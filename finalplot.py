import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

with open("forecasting.json", "r") as file:
    data = json.load(file)

a_saliva, a_stool, b_stool, m3_saliva, m3_stool, f4_saliva, f4_stool = [i for i in range(0,10)], [i for i in range(0,10)], [i for i in range(0,10)], [i for i in range(0,10)], [i for i in range(0,10)], [i for i in range(0,10)], [i for i in range(0,10)]
for key, value in data.items():
    try:
        subject = key.split(" ")[1]
        Biomaterial = key.split(" ")[2].split("-")[0]
        index = int(key.split(" ")[2].split("-")[1])-1
        if Biomaterial == "Saliva":
            if subject == "A":
                a_saliva[index] = float(value)
            elif subject == "M3":
                m3_saliva[index] = float(value)
            elif subject == "F4":
                f4_saliva[index] = float(value)
        elif Biomaterial == "Stool":
            if subject == "A":
                a_stool[index] = float(value)
            elif subject == "B":
                b_stool[index] = float(value)
            elif subject == "M3":
                m3_stool[index] = float(value)
            elif subject == "F4":
                f4_stool[index] = float(value)
    except: 
        pass

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(range(len(a_saliva)), a_saliva, edgecolors='#343795', color='#9900FF', marker='o', label='saliva - A', alpha=0.8, zorder=2, s=120)
ax.scatter(range(len(a_stool)), a_stool, edgecolors='#343795', color='#9900FF', marker='^', label='stool - A', alpha=0.8, zorder=2, s=120)

ax.scatter(range(len(b_stool)), b_stool, edgecolors='#5C618D', color='#619CFF', marker='^', label='stool - B', alpha=0.8, zorder=2, s=120)

ax.scatter(range(len(m3_saliva)), m3_saliva, edgecolors='#E8272A', color='#DA5819', marker='o', label='saliva - M3', alpha=0.8, zorder=2, s=120)
ax.scatter(range(len(m3_stool)), m3_stool, edgecolors='#E8272A', color='#DA5819', marker='^', label='stool - M3', alpha=0.8, zorder=2, s=120)

ax.scatter(range(len(f4_saliva)), f4_saliva, edgecolors='#458AC7', color='#03A09C', marker='o', label='saliva - F4', alpha=0.8, zorder=2, s=120)
ax.scatter(range(len(f4_stool)), f4_stool, edgecolors='#458AC7', color='#03A09C', marker='^', label='stool - F4', alpha=0.8, zorder=2, s=120)

biomaterial_patch = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='saliva'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10, label='stool')
]

subject_patch = [
    Line2D([0], [0], marker='.', color='w', markerfacecolor='#9900FF', markersize=10, label='A'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='#619CFF', markersize=10, label='B'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='#DA5819', markersize=10, label='M3'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='#03A09C', markersize=10, label='F4')
]

biomaterial_legend = ax.legend(handles=biomaterial_patch, title='Biomaterial', frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
subject_legend = ax.legend(handles=subject_patch, title='Subject', frameon=False, loc='upper left', bbox_to_anchor=(1, 0.85))

ax.add_artist(biomaterial_legend)
ax.add_artist(subject_legend)

ax.set_xlabel('Forecast step')
ax.set_ylabel('MAE')
ax.set_title('MAE w.r.t. Forecast step')

plt.tight_layout(rect=[0, 0, 0.9, 1])
fig.savefig('mae_forecast_step.png', dpi=300)