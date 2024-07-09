from pprint import pprint


def find_num_counts(nums):
    pair_dict = {}
    for num in nums:
        if num in pair_dict.keys():
            pair_dict[num] = pair_dict[num] + 1
        else:
            pair_dict[num] = 1
    final_pair = {}
    for key, value in pair_dict.items():
        if value > 1:
            final_pair[key] = value
    return pair_dict, final_pair


if __name__ == '__main__':
    nums = (1, 2, 3, 4, 5, 2, 3, 2, 3, 4, 6, 7, 2, 3, 4, 8, 5, 9)
    pair_dict, final_pair = find_num_counts(nums)
    pprint(pair_dict)
    pprint(final_pair)