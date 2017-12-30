import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';
import { Progress } from 'antd';
import Typography from 'material-ui/Typography';
import Card, { CardContent, CardMedia } from 'material-ui/Card';
import Grid from 'material-ui/Grid';

const styles = theme => ({
    root: theme.mixins.gutters({
      paddingTop: 16,
      paddingBottom: 16,
      marginTop: theme.spacing.unit * 3,
      flexGrow: 1,
    }),
    card: {
        padding: 20,
        marginBottom: 5,
        fontSize: '0.8em'
    }
  });


class Results extends React.Component {

    render() {
    const { result, classes, message } = this.props;
    let status = result.is_spam == 0 ? 'exception' : 'success';
    if(result.prob < 0.7 && result.prob > 0.3){
        status = 'active'
    }
    return (
        <div style={{marginTop: 20, width: '100%'}}>
            <Card className={classes.card}>
            {message}
            </Card>
            <h3> 判断结果 </h3>
            <div style={{width: '100%'}}>
            <Grid container>
                <Grid item xs={4} sm={6}>
                    <Progress
                        type="dashboard"
                        percent={result.prob.toFixed(2)*100}
                        status={status}
                    />
                </Grid>
                <Grid item xs={6}>
                    <CardContent>
                        <Typography>该短信为</Typography>
                        <Typography type="display1" color="secondary">
                        {result.is_spam === 0
                            ? <span style={{color: 'red'}}> 垃圾短信 </span>
                            : <span style={{color: 'green'}}>正常短信</span>
                            
                        }
                        </Typography>
                    </CardContent>  
                </Grid>
            </Grid>
            </div>
        </div>
    );

    }

}

export default withStyles(styles)(Results);
