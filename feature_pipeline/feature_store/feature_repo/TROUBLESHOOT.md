# Troubleshoot

## Can retrieve the historical batch features but for current time batch values are null

Possible root cause: the current timestamp > event_timestamp + ttl

TTL defines the expiry date of a feature value, after that date it's no longer used.
