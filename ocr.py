import re


def read_report(name):
    list_out_of_range = []
    regex = '^[a-z,A-Z ]*.(\d.*)'
    regex1 = '\).(\d.*$)'
    with open(f'reports/{name}.txt', 'r') as f:
        report = f.readlines()

    for i in report:
        if i == '\n':
            continue

        i = re.sub(r'\s{2}', r' ', i)
        i = re.sub(r':', r'', i)
        name = re.findall(regex, i)
        name_2 = re.findall(regex1, i)

        # print(i, name, name_2)
        if name != []:
            value = name[0].split(' ')[0]
            condition = name[0].split(' ')[1]
            if '-' in condition:
                down, up = condition.split('-')
                # print('name :', value, down, up)
                if float(value) <= float(down) or float(value) >= float(up):
                    list_out_of_range.append(i)

        if name_2 != []:
            value_2 = name_2[0].split(' ')[0]
            condition_2 = name_2[0].split(' ')[1]

            if '<' in condition_2:
                if float(value_2) >= float(condition_2.split('<')[1]):
                    list_out_of_range.append(i)
            if '-' in condition_2:
                down_2, up_2 = condition_2.split('-')
                # print('name 2:', value_2, down_2, up_2)
                if float(value_2) <= float(down_2) or float(value_2) >= float(up_2):
                    list_out_of_range.append(i)

    return list_out_of_range
