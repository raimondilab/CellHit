
def generate_prompt(template_path,**kwargs):
    
    # Read the template from the file
    with open(template_path, 'r') as file:
        template = file.read()

    # Replace placeholders in the template with actual values
    prompt = template.format(**kwargs)

    return prompt.strip()