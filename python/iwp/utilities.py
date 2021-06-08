def parse_range( range_string ):
    """
    Parse a range object from a string of the form:

      <start>:<stop>[:<step>]

    No validation is performed on <start>, <stop>, <step>.

    Takes 1 argument:

      range_string -

    Returns 1 value:

      range_object - range() object.

    """

    components = list( map( int, range_string.split( ":" ) ) )

    if len( components ) == 1:
        components = [components[0], components[0]]

    # adjust the end so it is inclusive rather than following Python's exclusive
    # semantics.  this makes things less surprising to end users.
    components[-1] += 1

    try:
        range_object = range( *components )
    except:
        range_object = None

    return range_object
