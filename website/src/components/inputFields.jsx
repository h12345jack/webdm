import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import MenuItem from 'material-ui/Menu/MenuItem';
import TextField from 'material-ui/TextField';
import HelpOutline from 'material-ui-icons/HelpOutline';
import Button from 'material-ui/Button';
import Radio, { RadioGroup } from 'material-ui/Radio';
import { FormLabel, FormControl, FormControlLabel, FormHelperText } from 'material-ui/Form';

const styles = theme => ({
    container: {
      display: 'flex',
      flexWrap: 'wrap',
      width: '100%'
    },
    textField: {
      marginLeft: theme.spacing.unit,
      marginRight: theme.spacing.unit,
    },
    menu: {
      width: '100%',
    },
  });

const Iconstyles ={
  style:{
    width: "1em",
    height: "1em",
  }
 
};

  
class TextFields extends React.Component {

    state = {
        model: '0',
        text : '',
        helpinfo : <div style={{textAlign:'right'}}><HelpOutline {...Iconstyles} /> {"复制粘贴提高效率"} </div>,
        submit_disable: false,
    }

    handleTextonChange = (e)=>{
      this.setState({
        text: e.target.value,
        helpinfo: ''
      })
    }

    handleSubmit = ()=>{
      const text = this.state.text;
      const helpinfo = <div style={{color: 'red', textAlign:'right'}}><HelpOutline {...Iconstyles} /> {"短信内容不能为空"} </div>;
      if(text === ''){
        this.setState({
          helpinfo
        })
      }else{
        this.setState({
          submit_disable: true
        })
        this.props.messageSubmit({
          message: this.state.text,
          model: this.state.model
        }).then((item)=>{
          this.setState({
            submit_disable: false
          })
        }).catch(err=>{
          console.error(err);
          this.setState({
            submit_disable: false
          })
        })
      }
    }

    handleModelChange = (e) => {
      const v = e.target ? e.target.value : e;
      this.setState({
        model: v
      })  
    }

    render() {
          
        const { classes } = this.props;
        return <form className={classes.container} noValidate autoComplete="off">
          
          <TextField
            id="full-width"
            label="输入你要识别的短信"
            InputLabelProps={{
              shrink: true,
            }}
            onChange = {this.handleTextonChange}
            placeholder="机器不想学习"
            helperText = {this.state.helpinfo}
            fullWidth
            rows={4}
            multiline
            margin="normal"
            value = {this.state.text}
          />
          <div style={{width: '100%'}}>
          <FormControl component="fieldset" required className={classes.formControl}>
          <FormLabel component="label" style={{fontSize: '0.75em'}}>模型选择</FormLabel>
          <RadioGroup
            aria-label="模型选择"
            name="gender1"
            className={classes.group}
            value={this.state.model}
            onChange={this.handleModelChange}
            row
          >
            <FormControlLabel value="0" control={<Radio />} label="LR" />
            <FormControlLabel value="1" control={<Radio />} label="SVM" />
            <FormControlLabel value="2" control={<Radio />} label="Naive Bayes" />
          </RadioGroup>
        </FormControl>
          </div>
          <div>
            <Button 
              raised 
              color="primary" 
              className={classes.button} 
              onClick={this.handleSubmit}
              disabled={this.state.submit_disable}
            >
              提交
            </Button>
          </div>
          </form>

    }

}

export default withStyles(styles)(TextFields);
