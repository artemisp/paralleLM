def qa_postproc_answer(answer):
    """
    format answer: {a}. 
    output: {a}
    """
    return answer.replace("answer:",'')