with open('/Users/daniel/PhD/spectral-comparison/code/simulations/specparam_knee_simulation.qmd', 'r') as f:
    lines = f.read().split('\n')

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if '# Setup the interactive sliders' in line:
        start_idx = i
        break

for i in range(start_idx, len(lines)):
    if 'interact(' in lines[i]:
        pass
    if '));' in lines[i]:
        end_idx = i
        break

new_code = """# Setup the interactive sliders dynamically
from ipywidgets import interactive, VBox, HBox, Output
import IPython.display

b_slider = FloatSlider(min=0.5, max=5.0, step=0.1, value=3.0, description='Offset (b)')
chi_slider = FloatSlider(min=0.5, max=4.0, step=0.1, value=1.5, description='Exponent (χ)')
k_slider = FloatSlider(min=0.1, max=1000.0, step=0.5, value=10.0, description='Knee (k)')

def update_k_range(*args):
    # Restrict max knee parameter so that f_k = k**(1/chi) <= 100 Hz
    # This means k <= 100**chi
    chi = chi_slider.value
    max_k = 100.0 ** chi
    k_slider.max = max_k
    if k_slider.value > max_k:
        k_slider.value = max_k

chi_slider.observe(update_k_range, 'value')
update_k_range() # Initialize constraints

interactive_plot = interactive(plot_interactive_knee, b=b_slider, k=k_slider, chi=chi_slider)
display(interactive_plot)"""

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx+1] = new_code.split('\n')
    with open('/Users/daniel/PhD/spectral-comparison/code/simulations/specparam_knee_simulation.qmd', 'w') as f:
        f.write('\n'.join(lines))
    print("Widget updated successfully.")
else:
    print("Could not find the widget setup code.")
