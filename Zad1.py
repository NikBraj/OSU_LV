def total_euro(workingHours, payRate):
    return workingHours * payRate

workingHours = input('Radni sati: ')
workingHours = workingHours.split(' ')[0]
workingHours = int(workingHours)

payRate = input('eura/h: ')
payRate = payRate.split(' ')[0]
payRate = float(payRate)

print(f'Ukupno: {total_euro(workingHours, payRate)}')

