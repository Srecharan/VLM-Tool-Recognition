"""
Pipeline Visualization Tool

This script generates a flowchart visualization of the VLM pipeline, showing
the progression from dataset preparation through model selection and approaches
to evaluation.

Author: akameswa, sselvam
"""
import graphviz


def create_pipeline_flowchart():
    """
    Creates and saves a visual flowchart of the VLM processing pipeline.
    
    The flowchart shows:
    1. Dataset preparation steps
    2. Model selection options
    3. Technical approaches (Fine-tuning and RAG)
    4. Evaluation methodology
    
    Returns:
        str: Path to the generated flowchart image
    """
    # Create a new Digraph object with improved styling
    dot = graphviz.Digraph(comment='VLM Pipeline Flowchart', format='png')
    dot.attr(rankdir='TB', size='8,6', dpi='300')
    # Remove default node attributes as we'll set them individually
    dot.attr('edge', fontname='Arial', fontsize='12', color='#333333', penwidth='1.5')
    dot.attr('graph', nodesep='1.0', ranksep='0.8', bgcolor='white')

    # --- Dataset Preparation ---
    dot.node('dataset', '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
      <TR><TD><B>Dataset Preparation</B></TD></TR>
      <TR><TD>Min ~20 images per tool</TD></TR>
      <TR><TD>1,000+ total images</TD></TR>
      <TR><TD>13 selected tool classes</TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded', color='#2E86C1', fillcolor='#D6EAF8', penwidth='2.0', fontname='Arial', fontsize='14', margin='0.3,0.2')

    # --- Model Selection ---
    dot.node('models', '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
      <TR><TD><B>Model Selection</B></TD></TR>
      <TR><TD>Qwen/ViT-7B-Instruct</TD></TR>
      <TR><TD>Phi-3-vision-128k</TD></TR>
      <TR><TD>Llama-3.2-11B-Vision</TD></TR>
      <TR><TD>SmolVLM</TD></TR>
      <TR><TD>PaliGemma</TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded', color='#8E44AD', fillcolor='#E8DAEF', penwidth='2.0', fontname='Arial', fontsize='14', margin='0.3,0.2')

    # --- Approaches node (hidden) ---
    dot.node('approaches', 'Approaches', shape='none', fontsize='16', fontname='Arial Bold')

    # --- Fine-tuning approach ---
    dot.node('finetune', '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
      <TR><TD><B>Fine-tuning Strategy</B></TD></TR>
      <TR><TD>Custom dataset training</TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded', color='#27AE60', fillcolor='#D5F5E3', penwidth='2.0', fontname='Arial', fontsize='14', margin='0.3,0.2')

    # --- RAG approach ---
    dot.node('rag', '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
      <TR><TD><B>RAG Approach</B></TD></TR>
      <TR><TD>Information retrieval from dataset</TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded', color='#E74C3C', fillcolor='#FADBD8', penwidth='2.0', fontname='Arial', fontsize='14', margin='0.3,0.2')

    # --- Evaluation ---
    dot.node('eval', '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
      <TR><TD><B>Evaluation</B></TD></TR>
      <TR><TD>Accuracy metrics</TD></TR>
    </TABLE>>''', shape='box', style='filled,rounded', color='#F39C12', fillcolor='#FCF3CF', penwidth='2.0', fontname='Arial', fontsize='14', margin='0.3,0.2')

    # --- Add edges to connect the pipeline stages with styled arrows ---
    dot.edge('dataset', 'models', color='#2E86C1', penwidth='1.8')
    dot.edge('models', 'approaches', color='#8E44AD', penwidth='1.8')
    dot.edge('approaches', 'finetune', color='#27AE60', penwidth='1.8')
    dot.edge('approaches', 'rag', color='#E74C3C', penwidth='1.8')
    dot.edge('finetune', 'eval', color='#27AE60', penwidth='1.8')
    dot.edge('rag', 'eval', color='#E74C3C', penwidth='1.8')

    # --- Add a subgraph to control layout ---
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('finetune')
        s.node('rag')

    # Save the graph to a file
    output_file = 'pipeline_flowchart'
    dot.render(output_file, view=False, cleanup=True)
    return f"{output_file}.png"


if __name__ == "__main__":
    output_path = create_pipeline_flowchart()
    print(f"Enhanced pipeline flowchart generated successfully as {output_path}")
    print("\nFlowchart shows the progression from:")
    print("1. Dataset preparation (~20 images per tool, 1,000+ total images)")
    print("2. Model selection (5 VLM candidates)")
    print("3. Two technical approaches: Fine-tuning and RAG")
    print("4. Evaluation methodology")
