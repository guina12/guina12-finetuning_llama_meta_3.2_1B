CRITERIA_EVAL = ["relevância","fatualidade"]

def get_criteria_eval(criteria):
    if criteria  in CRITERIA_EVAL:
        return criteria
    else:
        raise ValueError()