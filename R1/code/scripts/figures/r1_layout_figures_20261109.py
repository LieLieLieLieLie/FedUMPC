"""Layout-only revision of Figures 5--10; all plotted records are unchanged."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import inspect
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import r1_final_figures as f
import r1_external_expanded_figures as expanded
import pandas as pd

ROOT=R1_ROOT
DEST=ROOT.parents[1]/'paper/IoTJ_FedUMPC/R1/figures/Experiment/R1_Layout20261109'
DATA=artifact_path('results_r1_external_expanded')
FRESH_FLEET=False

def save_bottom_trimmed(fig,path):
    """Remove unused bottom canvas without rescaling fonts or panel widths."""
    fig.canvas.draw()
    content=fig.get_tightbbox(fig.canvas.get_renderer())
    crop=Bbox.from_extents(0,max(0,content.y0-.015),fig.get_figwidth(),fig.get_figheight())
    fig.savefig(path,format='pdf',bbox_inches=crop,pad_inches=0)

def save_compact(fig,name,*,top=.94,bottom=.11,hspace=.45,wspace=.38):
    right=.83 if name in {'fig7_federation.pdf','fig9_deployment_sensitivity.pdf','fig10_uncertainty_error.pdf'} else .98
    fig.subplots_adjust(left=.15,right=right,top=top,bottom=bottom,hspace=hspace,wspace=wspace)
    save_bottom_trimmed(fig,DEST/name)
    plt.close(fig)

def execute(function,replacements):
    source=inspect.getsource(function)
    for old,new in replacements:
        assert old in source,(function.__name__,old)
        source=source.replace(old,new)
    namespace=dict(function.__globals__)
    exec(compile(source,'<layout-20261109>','exec'),namespace)
    return namespace[function.__name__]

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    f.OUT=DEST;f.setup_style();f.save=save_compact
    f.FINAL=DATA
    for function,old_height,new_height,old_hspace in [
        (f.common_plant_figure,'4.7','4.2','0.62'),
        (f.mismatch_figure,'4.8','4.3','0.65')]:
        execute(function,[(f'figsize=(3.5, {old_height})',f'figsize=(3.5, {new_height})'),
                          (f'hspace={old_hspace}','hspace=0.50'),('ncol=3','ncol=4'),
                          ('columnspacing=0.9','columnspacing=0.55')] if function==f.common_plant_figure else
                         [(f'figsize=(3.5, {old_height})',f'figsize=(3.5, {new_height})'),
                          (f'hspace={old_hspace}','hspace=0.50'),('ncol=3','ncol=4'),
                          ('columnspacing=0.7','columnspacing=0.55')])()
    f.FINAL=artifact_path('results_r1_final_screened')
    if not FRESH_FLEET:
        execute(f.federation_figure,[('figsize=(3.5, 4.5)','figsize=(3.5, 4.1)'),
                                     ('hspace=0.65','hspace=0.50')])()
    execute(f.deployment_figure,[('figsize=(3.5, 4.5)','figsize=(3.5, 4.1)'),
                                 ('hspace=0.65','hspace=0.50'),('ncol=2','ncol=4'),
                                 ('"Repeatability"','"MC variance"'),('"Scoring time"','"Timing"'),
                                 ('bbox_to_anchor=(0.52, 0.005)','bbox_to_anchor=(0.52, 0.035)'),
                                 ('columnspacing=0.8, handlelength=1.4','columnspacing=0.55, handlelength=1.0'),
                                 ('bottom=0.19','bottom=0.19')])()
    execute(f.uncertainty_figure,[('figsize=(3.5, 4.6)','figsize=(3.5, 4.2)'),
                                  ('hspace=0.57','hspace=0.50')])()
    expanded.OUT=DEST
    expanded.save_bottom_trimmed=save_bottom_trimmed
    robust=execute(expanded.robustness_figure,[
        ('figsize=(3.5,5.2)','figsize=(3.5,4.3)'),
        ('hspace=.70','hspace=.38'),
        ("fig.savefig(OUT/'fig8_robustness.pdf',format='pdf')","save_bottom_trimmed(fig,OUT/'fig8_robustness.pdf')"),
        ("ax.set_ylabel(label,labelpad=1);f.panel(ax,letter,title)",
         "ax.set_ylabel(label,labelpad=1);f.panel(ax,letter,title)\n  if m=='MinClearance':\n   ax.tick_params(axis='y',labelrotation=55,pad=1)\n   ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))")])
    robust(pd.read_csv(DATA/'eight_condition_summary.csv'),pd.read_csv(DATA/'eight_condition_runs.csv'))
    print('LAYOUT_FIGURES_COMPLETE',DEST)

if __name__=='__main__':main()
