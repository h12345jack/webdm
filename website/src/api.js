
import axios from 'axios';

export function spamClassify(item){
    const params={
        message: item.message,
        model: item.model
    }
    return axios({
          method: 'post',
          url: 'http://127.0.0.1:5000/api/message',
          data: params
      })
}