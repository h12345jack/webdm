import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';

import Snackbar from 'material-ui/Snackbar';
import Fade from 'material-ui/transitions/Fade';
import { LinearProgress } from 'material-ui/Progress';

import TextFields from '../components/inputFields';
import Result from '../components/result';

import { spamClassify } from '../api.js';

const styles = theme => ({
    container: {
      display: 'flex',
      flexWrap: 'wrap',
      marginLeft: '5%',
      marginRight: '5%',
      marginTop: 20,
      marginBottom: 50,
    },
    textField: {
      marginLeft: theme.spacing.unit,
      marginRight: theme.spacing.unit,
    },
    menu: {
    },
    loading: {
      width: '100%',
      position: 'fixed',
      top: 1,
      left: 0
    }
  });

const Iconstyles ={
  style:{
    width: "1em",
    height: "1em",
  }
 
};

  
class Main extends React.Component {

  state = {
    loading: false,
    result: {}, 
    message: '',
    error: false
  }
  
  handleErrorOpen = () => {
    this.setState({ error: true });
  };

  handleErrorClose = () => {
    this.setState({ error: false });
  };

  messageSubmit = (item)=>{
    this.setState({
      loading: true,
      message: item.message,
      result:{}
    })
      
    return new Promise((resolve, reject) => {
      spamClassify({
        message: item.message,
        model: item.model
      }).then(res=>{
        this.setState({
          loading: false,
          result: res.data,
        })
        resolve(res)
      }).catch(err=>{
        this.setState({
          loading: false,
          result: {},
          error: true
        });
        reject(err);
      })
    })
  }
    render() {
        const { classes } = this.props;
        const { result, message } = this.state;
        console.log(result, 74)
        return <div className={classes.container} >
            {this.state.loading && <div className={classes.loading}><LinearProgress /></div>}
            <h3>在线实例</h3>
            <TextFields messageSubmit={this.messageSubmit}/>
            { (result.is_spam || result.is_spam==0) && <Result result={result} message={message}/>}

            <Snackbar
              anchorOrigin = {{ vertical: 'top', horizontal: 'center' }}
              open={this.state.error}
              onClose={this.handleErrorClose}
              transition={Fade}
              SnackbarContentProps={{
                'aria-describedby': 'message-id',
              }}
              message={<span id="message-id">生活难免错误，代码也是。</span>}
            />
         </div>

    }

}

export default withStyles(styles)(Main);
