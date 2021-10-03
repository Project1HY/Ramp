# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def bad_value(value, explanation=None):
    """
    Raise ValueError.  Useful when doing conditional assignment.
    e.g.
    dutch_hand = 'links' if eng_hand=='left' else 'rechts' if eng_hand=='right' else bad_value(eng_hand)
    """
    raise ValueError(f'Bad Value: {value}{": " + explanation if explanation is not None else ""}')


def assert_option(choice, possibilities):
    assert choice in possibilities, '"{}" was not in the list of possible choices: {}'.format(choice, possibilities)
