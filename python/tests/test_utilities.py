#!/usr/bin/env python3

# Tests for the utilities module.

import pytest

import iwp.utilities

class TestIndiciesToRegion:
    """
    Test harness for the utilities.indices_to_regions() method.  Verifies that
    it handles sorted indices, unsorted indices, and unsorted indices with
    duplicates.
    """

    def test_sorted_indices( self ):
        """
        Verifies that sorted inputs, potentially empty, can be converted to regions.

        Takes no arguments.

        Returns nothing.

        """

        test_cases = [
            # empty list.
            [[],                                 []],

            # example from help.  mixture of different sized regions (singletons
            # and larger) as well as indices that are very similar to the
            # positions within the input so as to expose logic errors.
            [[1, 2, 3, 5, 6, 8, 10, 11, 12, 13], [[1, 3], [5, 6], [8, 8], [10, 13]]],

            # multiple singletons.
            [[1, 3, 5, 7],                       [[1, 1], [3, 3], [5, 5], [7, 7]]],

            # two ranges.
            [[1, 2, 3, 4, 97, 98, 99, 100],      [[1, 4], [97, 100]]],
        ]

        for test_input, test_output in test_cases:
            assert test_output == iwp.utilities.indices_to_regions( test_input,
                                                                    is_sorted_flag=True )

    def test_unsorted_indices( self ):
        """
        Verifies that unsorted inputs, potentially empty, can be converted to regions.

        Takes no arguments.

        Returns nothing.

        """

        #
        # NOTE: derived from test_sorted_indices().  see its test cases for more
        #       details.
        #
        test_cases = [
            # inputs that are already sorted should still work.
            [[],                                 []],
            [[1, 2, 3, 5, 6, 8, 10, 11, 12, 13], [[1, 3], [5, 6], [8, 8], [10, 13]]],
            [[1, 3, 5, 7],                       [[1, 1], [3, 3], [5, 5], [7, 7]]],
            [[1, 2, 3, 4, 97, 98, 99, 100],      [[1, 4], [97, 100]]],

            # unsorted inputs.  these are permutations of the above sorted
            # versions.
            [[3, 2, 1, 8, 5, 6, 13, 10, 12, 11], [[1, 3], [5, 6], [8, 8], [10, 13]]],
            [[7, 1, 3, 5],                       [[1, 1], [3, 3], [5, 5], [7, 7]]],
            [[1, 100, 98, 4, 97, 2, 99, 3],      [[1, 4], [97, 100]]]
        ]

        for test_input, test_output in test_cases:
            assert test_output == iwp.utilities.indices_to_regions( test_input,
                                                                    is_sorted_flag=False )

    def test_duplicate_indices( self ):
        """
        Verifies that unsorted inputs with duplicates can be converted to regions.

        NOTE: We only test the unsorted interface as the implementation does not
              guarantee correctness with duplicates unless sorted.

        Takes no arguments.

        Returns nothing.

        """

        unsorted_test_cases = [
            # three regions with repetitions in ascending/descending patterns
            # and in interleaved order.
            [[1, 2, 3, 3, 2, 1, 6, 6, 1, 8, 1, 6], [[1, 3], [6, 6], [8, 8]]],

            # singleton.
            [[1, 1, 1, 1, 1, 1],                   [[1, 1]]],

            # two regions, interleaved in descending order.
            [[1, 99, 2, 3, 4, 97, 98, 99, 4, 100], [[1, 4], [97, 100]]],
        ]

        # verify we get the same output regardless of the sort order.
        for test_input, test_output in unsorted_test_cases:
            assert test_output == iwp.utilities.indices_to_regions( test_input,
                                                                    is_sorted_flag=False )


if __name__ == "__main__":
    pytest.main()
