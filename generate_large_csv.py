import csv

with open('test-data.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

with open('test-data-large.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for _ in range(50):
        writer.writerows(rows)
