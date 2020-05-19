from twilio.rest import Client
import os

class trainNotification:

    def sendsms():
        account_sid = 'ACb71e98ca51d84cb554ba2baec75b8ac3'
        auth_token = '2fe6c2f36fdaac2f44388b5af7132944'
        client = Client(account_sid, auth_token)
        message = client.messages \
            .create(
                 body='The Model Is Done Training',
                 from_='+12058787689',
                 to='+18182203266'
             )

        print(message.sid)
