def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    print(arrivals.index(name), len(arrivals))
    return (arrivals.index(name) >= len(arrivals)/2) and (arrivals.index(name) < len(arrivals)+1)
    

arrivals = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
name='Ford'

fashionably_late(arrivals, name)