import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';

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
      marginTop: 20
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
  }

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
         </div>

    }

}

export default withStyles(styles)(Main);
