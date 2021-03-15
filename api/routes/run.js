var express = require('express');
var router = express.Router();
//var app = express(); 

/*
router.get('/', function(req, res, next) {
    res.send('API is working properly');
});*/

/*app.listen(9000, function() { 
    console.log('server running on port 9000'); 
} ) */
  
// Function callName() is executed whenever  
// url is of the form localhost:3000/name 
router.get('/', callName); 
  
function callName(req, res) { 
      
    // Use child_process.spawn method from  
    // child_process module and assign it 
    // to variable spawn 
    var spawn = require("child_process").spawn; 
      
    // Parameters passed in spawn - 
    // 1. type_of_script 
    // 2. list containing Path of the script 
    //    and arguments for the script  
      
    // E.g : http://localhost:3000/name?firstname=Mike&lastname=Will 
    // so, first name = Mike and last name = Will 
    var process = spawn('python',["./model/hello.py"/*, 
                            req.query.firstname, 
req.query.lastname*/] ); 
  
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object 
    process.stdout.on('data', function(data) { 
        return res.send(data.toString()); 
    } ) 

    process.on('close', (code) => {
        if (code > 0){
        console.log(code);
          return res.status(500).json({ msg: "Internal server error" });
    }});
} 

module.exports = router;