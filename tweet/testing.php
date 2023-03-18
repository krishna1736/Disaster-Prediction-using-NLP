<?php
define("PG_DB"  , "postgres");
define("PG_HOST", "localhost");
define("PG_USER", "postgres");
define("PG_PORT", "5433");
define("PG_PWD", "krishna1736");


$pgcon = pg_connect("dbname=".PG_DB." host=".PG_HOST." user=".PG_USER." password=".PG_PWD." port=".PG_PORT);

if (!$pgcon)
{
echo "Not connected : " . pg_error();
exit;
}


$sql4="SELECT json_build_object('type', 'FeatureCollection','features', json_agg(ST_AsGeoJSON(t.*)::json))FROM ( select id,name ,coordinate as geom,tweet from testing_table) as t(id,name,geom,tweet);";
$query4 = pg_query($pgcon, $sql4);
$row4 = pg_fetch_array($query4) ;
echo $row4 [0];


?>
      