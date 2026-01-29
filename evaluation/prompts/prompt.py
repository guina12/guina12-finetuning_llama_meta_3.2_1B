
#--------------------------------------------------------------------------------------------------------------------------------#

def prompt_instruction(criteria, question_user, model_output):
    prompt = f"""
        Você é um juiz especialista. Avalie a qualidade de uma resposta fornecida
        para uma instrução com base nos critérios abaixo.

        1. {criteria}:
        Quão correta é, do ponto de vista factual, a informação apresentada na resposta?
        Você é um especialista técnico neste tema.

        2. Estilo:
        O tom e o estilo de escrita são apropriados para um post de blog ou conteúdo
        de redes sociais? O texto deve usar linguagem simples, porém técnica quando
        necessário, e evitar linguagem formal ou acadêmica.

        3. Escalas de Avaliação:
        Utilize escalas Likert de três pontos, conforme definido abaixo.

        Escala de Acurácia:
        1 (Ruim): Contém erros factuais ou informações enganosas
        2 (Boa): Majoritariamente correta, com pequenos erros ou omissões
        3 (Excelente): Altamente correta e abrangente

        Escala de Estilo:
        1 (Ruim): Muito formal, utiliza palavras excessivamente complexas
        2 (Bom): Bom equilíbrio entre conteúdo técnico e acessibilidade,
                mas ainda usa termos e expressões formais
        3 (Excelente): Linguagem perfeitamente acessível para blog/redes sociais,
                    usa termos técnicos simples e precisos apenas quando necessário

        Instrução:
        {question_user}

        Resposta:
        {model_output}

        Forneça sua avaliação em formato JSON, seguindo exatamente a estrutura abaixo:

        {{
        "accuracy": {{
            "analysis": "...",
            "score": 0
        }},
        "style": {{
            "analysis": "...",
            "score": 0
        }}
        }}
        """
    return prompt
