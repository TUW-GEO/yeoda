import datetime


def str2date(s):
    return datetime.datetime.strptime(s, '%Y').date()


def date2str(t):
    return t.strftime('%Y') if isinstance(t, datetime.date) else t
