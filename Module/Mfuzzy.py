

class membership_function:

    def linear_turun(dt, domain_a, domain_b):
        data = dt
        if data >= domain_a and data <= domain_b:
            hit_dt = (domain_b - data)/float(domain_b - domain_a)
        else:
            hit_dt = 'No Value'
        return hit_dt

    def linear_naik(dt, domain_a, domain_b):
        data = dt
        if data >= domain_a and data <= domain_b:
            hit_dt = (data - domain_a)/float(domain_b - domain_a)
        else:
            hit_dt = 'No Value'
        return hit_dt

    def fk_segitiga(dt, domain_a, domain_b, domain_c):
        data = dt
        if data <= domain_a and data >= domain_c:
            hit_dt = 0.00
        elif data >= domain_a and data <= domain_b:
            hit_dt = (data - domain_a)/float(domain_b - domain_a)
        elif data >= domain_b and data <= domain_c:
            hit_dt = (domain_c - data)/float(domain_c - domain_b)
        else:
            hit_dt = 'No Value'
        return hit_dt

    def fk_trapesium(dt, domain_a, domain_b, domain_c, domain_d):
        data = dt
        if data <= domain_a and data >= domain_d:
            hit_dt = 0.00
        elif data >= domain_a and data <= domain_b:
            hit_dt = (data - domain_a)/float(domain_b - domain_a)
        elif data >= domain_b and data <= domain_c:
            hit_dt = 1.00
        elif data >= domain_c and data <= domain_d:
            hit_dt = (domain_d - data)/float(domain_d - domain_c)
        else:
            hit_dt = 'No Value'
        return hit_dt