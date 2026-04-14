import traceback
from main import MarkerWrapper

m = MarkerWrapper(output_dir="./conversion_results")
try:
    m.convert_single("./input_equation/document_equations.pdf")
except Exception as e:
    traceback.print_exc()
