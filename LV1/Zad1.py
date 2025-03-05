def total_euro(workingHours, payRate):
    return workingHours * payRate

while True:

    workingHours = input('Radni sati: ')
    workingHours = workingHours.split(' ')[0]
    
    try: 
        workingHours = float(workingHours)
        break

    except ValueError:
        print("To nije ispravna vrijednost, pokusajte ponovno.")


while True:

    payRate = input('eura/h: ')
    payRate = payRate.split(' ')[0]
    
    
    try: 
        payRate = float(payRate)
        break

    except ValueError:
        print("To nije ispravna vrijednost, pokusajte ponovno.")

    

print(f'Ukupno: {total_euro(workingHours, payRate)}')

